# adapted from Distill-VQ
import os.path as osp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import faiss
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment


class ProductQuantization(nn.Module):
    def __init__(
        self,
        pq_type: str = 'pq',
        subvector_num: int = 32,
        subvector_bits: int = 8,
        dist_mode: str = 'ip',
        emb_size: int = 768,
        pq_init_method: str = 'faiss',
        pq_update_method: str = 'grad',
        tie_nci_pq_centroid: int = 0,
        lm_head: nn.Parameter = None,
        centroid_update_loss: str = 'none',
        rq_topk_score: str = 'prod',
    ):
        super().__init__()
        assert pq_type in ('pq', 'opq', 'rq')
        assert dist_mode in ('ip', 'l2', 'iptol2')
        # dist_mode 'iptol2' means using l2 to represent ip
        # https://gist.github.com/mdouze/e4bdb404dbd976c83fe447e529e5c9dc
        # query (q1,...,qd) to (q1,...,qd,0)
        # document (p1,...,pd) to (p1,...,pd,sqrt(PHI-l2norm^2))
        self.pq_type = pq_type
        self.subvector_num = subvector_num
        self.subvector_bits = subvector_bits
        self.subvector_cents = 2 ** subvector_bits
        self.dist_mode = dist_mode
        self.emb_size = emb_size
        self.pq_init_method = pq_init_method
        self.pq_update_method = pq_update_method
        self.tie_nci_pq_centroid = tie_nci_pq_centroid
        self.centroid_update_loss = centroid_update_loss
        self.rq_topk_score = rq_topk_score
        self.get_preds = False

        if self.pq_type == 'rq':
            last_dim = emb_size
        else:
            last_dim = emb_size // subvector_num
        self.last_dim = last_dim

        if self.tie_nci_pq_centroid:
            assert self.dist_mode != 'iptol2'
            assert self.pq_type != 'opq'
            self.lm_head = lm_head
            if self.pq_type == 'pq':
                self.weight_layers = nn.Sequential(
                    nn.Linear(emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, last_dim))
        else:
            codebook_last_dim = last_dim
            if self.dist_mode == 'iptol2':
                codebook_last_dim += 1
            self.codebook = nn.Parameter(torch.empty(
                subvector_num, self.subvector_cents, codebook_last_dim), requires_grad=(self.pq_update_method == 'grad')).type(torch.FloatTensor)
            if self.pq_type == 'opq':
                self.rotate = nn.Parameter(torch.empty(
                    emb_size, emb_size), requires_grad=False).type(torch.FloatTensor)
            if self.pq_update_method == 'ema':
                # adapted from rq-vae-transformer
                self.decay = 0.99
                self.eps = 1e-5
                self.restart_unused_codes = True
                self.register_buffer('cluster_size_ema', torch.zeros(
                    subvector_num, self.subvector_cents))
                self.register_buffer(
                    'embed_ema', self.codebook.detach().clone())

    def augment_xb(self, xb, phi=None):
        norms = np.sum((xb ** 2), axis=-1)
        if phi is None:
            phi = np.max(norms)
        extracol = np.sqrt(phi - norms)
        return np.hstack((xb, extracol[..., np.newaxis]))

    def augment_xq(self, xq):
        if isinstance(xq, torch.Tensor):
            xq = torch.cat((xq, xq.new_zeros(*xq.shape[:-1], 1)), dim=-1)
        else:
            xq = np.concatenate(
                (xq, np.zeros((*xq.shape[:-1], 1), dtype=xq.dtype)), axis=-1)
        return xq

    def wrapped_augment_xb(self, xb, index=None):
        if self.dist_mode != 'iptol2':
            return xb
        emb_size = self.last_dim
        reshaped = False
        if xb.shape[-1] not in (emb_size, emb_size + 1):
            ori_shape = xb.shape
            xb = xb.reshape(-1, emb_size)
            reshaped = True
        if xb.shape[-1] == emb_size:
            xb = self.augment_xb(xb)
            extracol = xb[..., -1]
            if reshaped:
                xb = xb.reshape(*ori_shape[:-1], -1)
        else:
            extracol = xb[..., -1]
        assert xb.shape[-1] == emb_size + 1
        if index is None:
            self.extracol[:] = torch.tensor(extracol, dtype=torch.float32)
        else:
            self.extracol[:, index] = torch.tensor(
                extracol, dtype=torch.float32)
        return xb

    def rq_minus_centroids(self, embeddings, centroids):
        return embeddings - centroids[..., :self.last_dim]

    def compute_scores(self, a, b):
        if self.dist_mode == 'ip':
            result = a * b
        else:
            if self.dist_mode == 'iptol2' and a.shape[-1] != b.shape[-1]:
                a = self.augment_xq(a)
            result = - ((a - b) ** 2)
        return torch.sum(result, dim=-1)

    def get_codebook(self):
        if self.tie_nci_pq_centroid:
            codebook = self.lm_head[2:2+self.subvector_num*self.subvector_cents].reshape(
                self.subvector_num, self.subvector_cents, self.emb_size)
            if self.pq_type == 'pq':
                codebook = self.weight_layers(codebook)
            return codebook
        else:
            return self.codebook

    def codebook_from_index(self, index, index_file=None):
        if index is None:
            index = faiss.read_index(index_file)
        if self.pq_type == 'opq':
            assert isinstance(index, faiss.IndexPreTransform)
            vt = faiss.downcast_VectorTransform(index.chain.at(0))
            assert isinstance(vt, faiss.LinearTransform)
            rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
            index = faiss.downcast_index(index.index)
        else:
            rotate = None
            index = index

        if self.pq_type == 'rq':
            rq = index.rq
            centroid_embeds = faiss.vector_to_array(rq.codebooks)
            codebook = centroid_embeds.reshape(
                rq.M, rq.total_codebook_size // rq.M, -1)
            assert self.last_dim == rq.d
        else:
            pq_index = index
            centroid_embeds = faiss.vector_to_array(pq_index.pq.centroids)
            codebook = centroid_embeds.reshape(
                pq_index.pq.M, pq_index.pq.ksub, -1)
            assert self.last_dim == pq_index.pq.dsub
        assert self.codebook.shape[-1] == codebook.shape[-1], 'Codebook shape not aligned! Considering whether iptol2.'

        with torch.no_grad():
            self.codebook.copy_(torch.tensor(codebook))
            if self.pq_type == 'opq':
                self.rotate.copy_(torch.tensor(rotate))

    def build_faiss_index(self, doc_embeddings, save_path=None):
        assert self.dist_mode in ('ip', 'l2', 'iptol2')
        index_metric = faiss.METRIC_INNER_PRODUCT if self.dist_mode == 'ip' else faiss.METRIC_L2

        if self.pq_type == 'pq':
            components = f"PQ{self.subvector_num}x{self.subvector_bits}"
        elif self.pq_type == 'opq':
            components = f"OPQ{self.subvector_num},PQ{self.subvector_num}x{self.subvector_bits}"
        elif self.pq_type == 'rq':
            components = f"RQ{self.subvector_num}x{self.subvector_bits}"
        if self.dist_mode == 'iptol2':
            assert self.pq_type != 'rq', 'Not support iptol2 rq using faiss index.'
            doc_embeddings = self.wrapped_augment_xb(doc_embeddings)
        emb_size = doc_embeddings.shape[-1]

        index = faiss.index_factory(
            emb_size, components, index_metric)

        index.train(doc_embeddings)
        index.add(doc_embeddings)

        if save_path is not None:
            faiss.write_index(index, save_path)
        return index

    @torch.no_grad()
    def get_document_cluster_simple(self, return_mapping: bool = False):
        assert self.get_preds
        cluster = defaultdict(list)
        mapping = {}
        for i, p in enumerate(self.last_preds):
            label = tuple(p.tolist())
            mapping[i] = label
            cluster[label].append(i)
        del self.last_preds
        self.get_preds = False
        if return_mapping:
            return dict(cluster), mapping
        else:
            return dict(cluster)

    @torch.no_grad()
    def get_document_cluster(self, doc_embeddings: np.ndarray, rank: int, nrank: int, batch_size: int = 1024, return_mapping: bool = False):
        num_docs = doc_embeddings.shape[0]
        num_docs_per_worker = num_docs // nrank
        start = num_docs_per_worker * rank
        if rank + 1 == nrank:
            ending = num_docs
        else:
            ending = start + num_docs_per_worker
        part_num_docs = ending - start
        cluster = torch.empty(
            (part_num_docs, self.subvector_num), dtype=torch.int32)
        if self.pq_type == 'rq':
            func = self.get_rq_document_cluster
        else:
            func = self.get_pq_document_cluster
        func(doc_embeddings, cluster, start,
             ending, rank, batch_size)
        if return_mapping:
            new_mapping = {}
        doc_cluster = defaultdict(list)
        for k, v in enumerate(cluster):
            tuplev = tuple(v.tolist())
            key = k + start
            doc_cluster[tuplev].append(key)
            if return_mapping:
                new_mapping[key] = tuplev
        print('Number of document clusters:', len(doc_cluster))
        if return_mapping:
            return dict(doc_cluster), new_mapping
        else:
            return dict(doc_cluster)

    def get_pq_document_cluster(self, doc_embeddings: np.ndarray, cluster: torch.Tensor, start: int, ending: int, rank: int, batch_size: int = 1024):
        emb_size = doc_embeddings.shape[1]
        part_emb_size = emb_size // self.subvector_num
        codebook = self.get_codebook().cpu()
        with torch.no_grad():
            for i in tqdm(range(start, ending, batch_size), desc=f'Document Cluster {rank}'):
                batch_start = i
                batch_ending = min(i+batch_size, ending)
                cur_embeddings = torch.tensor(
                    doc_embeddings[batch_start:batch_ending], device='cpu')
                if self.pq_type == 'opq':
                    cur_embeddings = torch.matmul(
                        cur_embeddings, self.rotate.cpu().T)
                cur_codebook = codebook.unsqueeze(
                    1).expand(-1, cur_embeddings.size(0), -1, -1)
                if self.dist_mode == 'iptol2':
                    cur_extracol = self.extracol[batch_start:batch_ending]
                for j in range(self.subvector_num):
                    emb_start = j*part_emb_size
                    emb_ending = emb_start + part_emb_size
                    part_doc_embeddings = cur_embeddings[:,
                                                         emb_start:emb_ending]
                    if self.dist_mode == 'iptol2':
                        part_doc_embeddings = torch.cat(
                            (part_doc_embeddings, cur_extracol[:, j].unsqueeze(-1)), -1)
                    part_doc_embeddings = part_doc_embeddings.unsqueeze(-2)
                    part_codebook = cur_codebook[j]
                    distance = self.compute_scores(
                        part_doc_embeddings, part_codebook)
                    index = distance.max(dim=-1, keepdim=True)[1].squeeze(-1)
                    cluster[batch_start - start: batch_ending - start, j] = index

    def get_rq_document_cluster(self, doc_embeddings: np.ndarray, cluster: torch.Tensor, start: int, ending: int, rank: int, batch_size: int = 1024):
        part_num_docs = ending - start
        temp_docemb = torch.tensor(doc_embeddings[start:ending], device='cpu')
        codebook = self.get_codebook()
        with torch.no_grad():
            for j in range(self.subvector_num):
                part_codebook = codebook[j].cpu()
                if self.dist_mode == 'iptol2':
                    cur_extracol = self.extracol[:, j].unsqueeze(-1)
                for i in tqdm(range(0, part_num_docs, batch_size), desc=f'Document Cluster {rank}-{j}'):
                    batch_start = i
                    batch_ending = min(i+batch_size, part_num_docs)
                    cur_embeddings = temp_docemb[batch_start:batch_ending]
                    if self.dist_mode == 'iptol2':
                        cur_embeddings = torch.cat(
                            (cur_embeddings, cur_extracol[batch_start:batch_ending]), -1)
                    cur_embeddings = cur_embeddings.unsqueeze(-2)
                    cur_codebook = part_codebook.unsqueeze(
                        0).expand(cur_embeddings.size(0), -1, -1)
                    distance = self.compute_scores(
                        cur_embeddings, cur_codebook)
                    index = distance.max(dim=-1, keepdim=True)[1].squeeze(-1)
                    cluster[batch_start:batch_ending, j] = index
                    temp_docemb[batch_start:batch_ending] = self.rq_minus_centroids(
                        temp_docemb[batch_start:batch_ending], part_codebook[index])

    def forward(self, vecs, return_loss=True):
        # TODO: check results and determine which is better
        # before commit we generate label AFTER gumbel softmax
        # after commit we generate label BEFORE gumbel softmax
        if self.pq_type == 'rq':
            proba, index, loss = self.forward_rq(vecs, return_loss)
        else:
            proba, index, loss = self.forward_pq(vecs, return_loss)
        if self.training and self.pq_update_method == 'ema':
            self.ema_update(vecs, index)
        return proba, index, loss

    def forward_pq(self, vecs, return_loss=True):
        if self.pq_type == 'opq':
            vecs = torch.matmul(vecs, self.rotate.T)
        vecs = vecs.view(vecs.size(0), self.subvector_num, -1)
        codebook = self.get_codebook().unsqueeze(0).expand(vecs.size(0), -1, -1, -1)
        proba = self.compute_scores(vecs.unsqueeze(-2), codebook)
        index = proba.max(dim=-1)[1]
        if return_loss:
            if self.centroid_update_loss == 'reconstruct':
                reconstruct_emb = self.get_reconstruct_vector(index, codebook).view(
                    index.shape[0], self.subvector_num, self.last_dim)
                loss = ((vecs - reconstruct_emb) ** 2).mean()
            else:
                loss = None
        else:
            loss = None
        return proba, index, loss

    def forward_rq(self, vecs, return_loss=True):
        allproba = []
        index = []
        codebook = self.get_codebook()
        use_reconstruct_loss = self.centroid_update_loss == 'reconstruct'
        if use_reconstruct_loss:
            errors = []
        for i in range(self.subvector_num):
            cur_codebook = codebook[i:i+1].expand(vecs.size(0), -1, -1)
            proba = self.compute_scores(vecs.unsqueeze(-2), cur_codebook)
            part_index = proba.max(dim=-1)[1]
            allproba.append(proba)
            index.append(part_index)
            if self.dist_mode == 'iptol2':
                cur_centroid = codebook[i][part_index][..., :-1]
            else:
                cur_centroid = codebook[i][part_index]
            if use_reconstruct_loss:
                cur_error = vecs.detach() - cur_centroid
                errors.append(cur_error)
            if i != self.subvector_num - 1:
                vecs -= cur_centroid.detach()
        proba = torch.stack(allproba, dim=1)
        index = torch.stack(index, dim=1)
        if return_loss:
            if use_reconstruct_loss:
                loss = (torch.stack(errors) ** 2).mean()
                # loss = (vecs ** 2).mean()
            else:
                loss = None
        else:
            loss = None
        return proba, index, loss

    @torch.no_grad()
    def ema_update(self, vectors, idxs):
        assert self.dist_mode != 'iptol2'
        embed_dim = self.last_dim
        if self.pq_type == 'rq':
            vectors = vectors.unsqueeze(
                1).expand(-1, self.subvector_num, -1).reshape(-1, embed_dim)
        else:
            if self.pq_type == 'opq':
                vectors = torch.matmul(vectors, self.rotate.T)
            vectors = vectors.reshape(-1, embed_dim)
        idxs = idxs.reshape(-1)
        n_vectors = vectors.shape[0]
        one_hot_idxs = vectors.new_zeros(n_vectors, self.subvector_cents)
        one_hot_idxs.scatter_(dim=1,
                              index=idxs.unsqueeze(1),
                              src=vectors.new_ones(n_vectors, 1)
                              )
        one_hot_idxs = one_hot_idxs.reshape(-1,
                                            self.subvector_num, self.subvector_cents)
        cluster_size = one_hot_idxs.sum(dim=0)
        vectors_sum_per_cluster = torch.bmm(one_hot_idxs.permute(
            1, 2, 0), vectors.reshape(-1, self.subvector_num, embed_dim).permute(1, 0, 2))

        if dist.is_initialized():
            dist.all_reduce(vectors_sum_per_cluster, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        self.cluster_size_ema.mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(
            vectors_sum_per_cluster, alpha=1 - self.decay)

        if self.restart_unused_codes:
            temp = vectors.reshape(-1, self.subvector_num, embed_dim)
            B = temp.shape[0]
            if B < self.subvector_cents:
                n_repeats = (self.subvector_cents + B - 1) // B
                std = temp.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
                temp = temp.repeat(n_repeats, 1, 1)
                temp = temp + torch.rand_like(temp) * std

            _vectors_random = torch.stack([temp[torch.randperm(
                temp.shape[0], device=temp.device), i][:self.subvector_cents] for i in range(self.subvector_num)])

            if dist.is_initialized():
                dist.broadcast(_vectors_random, 0)

            usage = (self.cluster_size_ema.unsqueeze(-1) >= 1).float()
            self.embed_ema.mul_(usage).add_(_vectors_random * (1-usage))
            usage = usage.squeeze(-1)
            self.cluster_size_ema.mul_(usage)
            self.cluster_size_ema.add_(torch.ones_like(
                self.cluster_size_ema) * (1-usage))

        # update embedding
        n = self.cluster_size_ema.sum(dim=1, keepdim=True)
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps) /
            (n + self.subvector_cents * self.eps)
        )
        self.codebook.data[:] = self.embed_ema / \
            normalized_cluster_size.unsqueeze(-1)

    def fix(self):
        self.codebook.requires_grad_(False)
        if self.pq_type == 'opq':
            self.rotate.requires_grad_(False)

    @torch.no_grad()
    def initialize(self, index_file, doc_emb, rank, seed, pq_cluster_path, encode_batch_size):
        if self.dist_mode == 'iptol2':
            self.extracol = torch.empty(
                (doc_emb.shape[0], self.subvector_num), dtype=torch.float32)
        if self.pq_init_method != 'none':
            use_file = index_file is not None and osp.isfile(index_file)
            if self.pq_init_method.endswith('kmeans') and not use_file:
                self.get_preds = True
            if rank == 0:
                if self.pq_init_method == 'faiss':
                    if use_file:
                        print("Intializing codebook with faiss index file...")
                        index = faiss.read_index(index_file)
                    else:
                        print("Intializing codebook by building faiss index...")
                        index = self.build_faiss_index(
                            doc_emb, save_path=index_file)
                    self.codebook_from_index(index)
                elif self.pq_init_method.endswith('kmeans'):
                    if use_file:
                        print("Intializing codebook with torch file...")
                        tensor = torch.load(index_file, map_location='cpu')
                        self.codebook.copy_(tensor)
                        del tensor
                    else:
                        print("Intializing codebook by kmeans clustering...")
                        self.unsupervised_update_codebook_manually(
                            doc_emb, seed, self.pq_init_method)
                        if index_file is not None:
                            torch.save(self.codebook, index_file)
                elif self.pq_init_method == 'avg':
                    if use_file:
                        tensor = torch.load(index_file, map_location='cpu')
                        self.codebook.copy_(tensor)
                        del tensor
                    else:
                        torch.nn.init.normal_(
                            self.codebook.data, mean=0.0, std=0.01)
                    print(
                        f"Intializing codebook using average document embedding after {use_file} use file...")
                    self.init_pq_using_document_cluster(
                        doc_emb, pq_cluster_path, encode_batch_size)
            if dist.is_initialized():
                dist.broadcast(self.codebook.data, 0)
                if self.pq_type == 'opq':
                    dist.broadcast(self.rotate.data, 0)

    @torch.no_grad()
    def init_pq_using_document_cluster(self, doc_emb, cluster, batch_size):
        assert self.dist_mode in ('l2', 'iptol2')
        assert self.pq_type in ('pq', 'rq')
        with open(cluster, 'rb') as fr:
            cluster = pickle.load(fr)
        if self.dist_mode == 'iptol2':
            doc_emb = self.wrapped_augment_xb(doc_emb)
        doc_emb = np.array(doc_emb)

        macro_cluster = [defaultdict(list) for _ in range(self.subvector_num)]
        for k, v in cluster.items():
            for i in range(self.subvector_num):
                macro_cluster[i][k[i]].extend(v)
        macro_cluster = [dict(mc) for mc in macro_cluster]
        if self.pq_type == 'rq':
            part_doc_emb = doc_emb
        for i in range(self.subvector_num):
            cur_cluster = macro_cluster[i]
            if self.pq_type == 'pq':
                dstart = i * self.last_dim
                dending = dstart + self.last_dim
                part_doc_emb = doc_emb[:, dstart:dending]
            cur_codebook = self.codebook[i]
            for k, v in tqdm(cur_cluster.items(), desc=f'Averaging Embedding'):
                accum = np.zeros(self.last_dim)
                ndocs = len(v)
                for start in range(0, ndocs, batch_size):
                    docs = v[start:start+batch_size]
                    cur_emb = part_doc_emb[docs]
                    accum += (np.sum(cur_emb, axis=0) / ndocs)
                cur_codebook[k][:] = torch.tensor(accum)
                if self.pq_type == 'rq' and i != self.subvector_num - 1:
                    for start in range(0, ndocs, batch_size):
                        docs = v[start:start+batch_size]
                        part_doc_emb[docs] -= accum
        # torch.save(self.codebook, 'newcodebook.pt')

    @torch.no_grad()
    def unsupervised_update_codebook(self, doc_emb, rank, seed, align=False):
        if self.pq_update_method.endswith('kmeans'):
            self.get_preds = True
        if rank == 0:
            if align:
                ori_codebook = self.codebook.data.clone()
            if self.pq_update_method == 'faiss':
                self.unsupervised_update_codebook_faiss(doc_emb, seed)
            elif self.pq_update_method.endswith('kmeans'):
                self.unsupervised_update_codebook_manually(
                    doc_emb, seed, self.pq_update_method)
            if align:
                self.align_codebook(ori_codebook)
        dist.broadcast(self.codebook.data, 0)
        if self.pq_type == 'opq':
            dist.broadcast(self.rotate.data, 0)

    @torch.no_grad()
    def unsupervised_update_codebook_faiss(self, doc_emb, seed):
        print("Updating codebook using Faiss...")
        index = self.build_faiss_index(doc_emb)
        self.codebook_from_index(index)

    @torch.no_grad()
    def unsupervised_update_codebook_manually(self, doc_emb, seed, kmeans_method):
        print("Updating codebook using KMeans...")
        assert self.pq_type != 'opq'
        emb_size = doc_emb.shape[1]
        k = 2 ** self.subvector_bits
        if kmeans_method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans
            if doc_emb.shape[0] >= 1e3:
                kmeans = MiniBatchKMeans(n_clusters=k, max_iter=300, n_init=100, init='k-means++', random_state=seed,
                                         batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)
            else:
                kmeans = KMeans(n_clusters=k, max_iter=300, n_init=100,
                                init='k-means++', random_state=seed, tol=1e-7)
        elif kmeans_method == 'balancekmeans':
            raise NotImplementedError
        else:
            assert False

        preds = []
        if self.pq_type == 'pq':
            part_emb_size = emb_size // self.subvector_num
            codebook = []
            for i in tqdm(range(self.subvector_num), 'KMeans Clustering'):
                start = i * part_emb_size
                ending = start + part_emb_size
                cur_docemb = doc_emb[:, start:ending]
                cur_docemb = self.wrapped_augment_xb(cur_docemb, i)
                pred = kmeans.fit_predict(cur_docemb)
                codebook.append(kmeans.cluster_centers_)
                preds.append(pred)
            codebook = np.stack(codebook)
        elif self.pq_type == 'rq':
            codebook = []
            temp_docemb = np.array(doc_emb)
            for i in tqdm(range(self.subvector_num), 'KMeans Clustering'):
                temp_docemb = self.wrapped_augment_xb(temp_docemb, i)
                pred = kmeans.fit_predict(temp_docemb)
                centers = kmeans.cluster_centers_
                codebook.append(centers)
                preds.append(pred)
                if i != self.subvector_num - 1:
                    temp_docemb = self.rq_minus_centroids(
                        temp_docemb[..., :self.last_dim], centers[pred])
            codebook = np.stack(codebook)
        preds = np.stack(preds, axis=-1)
        self.last_preds = preds
        with torch.no_grad():
            self.codebook.copy_(torch.tensor(codebook))

    @torch.no_grad()
    def align_codebook(self, ori_codebook):
        # use sort; only need to sort 256^2 numbers using 8 bits
        new_codebook = self.codebook.new(*self.codebook.shape)
        for ori, cur, new in zip(ori_codebook, self.codebook, new_codebook):
            scores = self.compute_scores(ori.unsqueeze(0), cur.unsqueeze(1))
            scores = scores.cpu().numpy()
            assign = linear_sum_assignment(scores, maximize=True)
            for cid, oid in zip(*assign):
                new[oid] = cur[cid]
        with torch.no_grad():
            self.codebook.copy_(new_codebook)

    @torch.no_grad()
    def beam_search(
        self,
        doc_emb: torch.Tensor,
        num_return_sequences,
        num_beams=None,
        do_sample=False,
        return_proba=False,
    ):
        if num_beams is None:
            num_beams = num_return_sequences

        # (nsub, vocab, dim)
        codebook = self.get_codebook()
        if doc_emb.device != codebook.device:
            doc_emb = doc_emb.to(codebook.device)
        if self.pq_type == 'opq':
            doc_emb = torch.matmul(doc_emb, self.rotate.T)
        batch_size = doc_emb.size(0)
        # (bs, beam)
        beam_scores = doc_emb.new_ones(batch_size, 1)
        if self.pq_type == 'rq':
            # (bs, beam, dim)
            temp_embed = doc_emb.unsqueeze(1).clone()
        # (bs, beam, nsub)
        temp_index = torch.zeros(
            (batch_size, 1, 1), device=doc_emb.device, dtype=torch.int32)
        # (beam * vocab)
        beam_indices = torch.div(torch.arange(
            num_beams * self.subvector_cents, device=doc_emb.device), self.subvector_cents, rounding_mode='floor')
        code_indices = torch.arange(
            self.subvector_cents, device=doc_emb.device).repeat(num_beams)

        for i in range(self.subvector_num):
            if self.pq_type == 'rq':
                # (bs, 1, vocab, dim)
                cur_codebook = codebook[i:i +
                                        1].expand(batch_size, -1, -1).unsqueeze(1)
                cur_embed = temp_embed.unsqueeze(-2)
            else:
                # (bs, vocab, dim)
                cur_codebook = codebook[i].unsqueeze(
                    0).expand(batch_size, -1, -1)
                start = i * self.last_dim
                ending = start + self.last_dim
                # (bs, dim)
                cur_embed = doc_emb[:, start:ending].unsqueeze(1)
            # (bs, beam, vocab)
            proba = self.compute_scores(cur_embed, cur_codebook)
            proba = F.softmax(proba, dim=-1)
            if self.pq_type == 'rq':
                if self.rq_topk_score == 'prod':
                    proba = beam_scores.unsqueeze(-1) * proba
                else:
                    proba = proba
            else:
                proba = beam_scores.unsqueeze(-1) * proba.unsqueeze(1)
            # (bs, beam * vocab)
            proba = proba.view(batch_size, -1)
            prev_nbeam = beam_scores.size(1)
            if prev_nbeam == num_beams:
                cur_beam_indices = beam_indices
                cur_code_indices = code_indices
            else:
                cur_beam_indices = torch.div(torch.arange(
                    prev_nbeam * self.subvector_cents, device=doc_emb.device), self.subvector_cents, rounding_mode='floor')
                cur_code_indices = torch.arange(
                    self.subvector_cents, device=doc_emb.device).repeat(prev_nbeam)
            ncandidate = proba.size(1)
            assert len(cur_beam_indices) == len(
                cur_code_indices) == ncandidate
            # (bs, beam)
            if num_beams < ncandidate:
                if do_sample:
                    topk_indices = torch.multinomial(
                        proba, num_samples=num_beams)
                else:
                    _, topk_indices = proba.topk(num_beams, dim=-1)
                topk_prev_beams = cur_beam_indices[topk_indices].unsqueeze(-1)
                topk_cur_code = cur_code_indices[topk_indices]
                beam_scores = proba.gather(1, topk_indices)
                # (bs, beam, nsub)
                temp_index = torch.cat([temp_index.gather(
                    1, topk_prev_beams.expand(-1, -1, temp_index.size(-1))), topk_cur_code.unsqueeze(-1)], dim=-1)
                if self.pq_type == 'rq' and i != self.subvector_num - 1:
                    # (bs, beam, dim)
                    temp_embed = temp_embed.gather(
                        1, topk_prev_beams.expand(-1, -1, temp_embed.size(-1))) - codebook[i][topk_cur_code][..., :self.last_dim]
            else:
                beam_scores = proba
                temp_index = torch.cat([temp_index.repeat_interleave(
                    self.subvector_cents, dim=1), cur_code_indices.unsqueeze(-1).unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
                if self.pq_type == 'rq' and i != self.subvector_num - 1:
                    temp_embed = temp_embed.repeat_interleave(
                        self.subvector_cents, dim=1) - codebook[i][cur_code_indices][..., :self.last_dim]
        assert beam_scores.size(1) == num_beams
        topk_label = temp_index[:, :, 1:]
        if return_proba:
            return topk_label, beam_scores
        else:
            return topk_label

    @torch.no_grad()
    def get_topk_document_mapping(
        self,
        doc_embeddings: np.ndarray,
        rank: int,
        nrank: int,
        num_return_sequences: int,
        batch_size: int = 1024,
    ):
        num_docs = doc_embeddings.shape[0]
        num_docs_per_worker = num_docs // nrank
        start = num_docs_per_worker * rank
        if rank + 1 == nrank:
            ending = num_docs
        else:
            ending = start + num_docs_per_worker
        part_num_docs = ending - start
        all_topk_label = torch.empty(
            (part_num_docs, num_return_sequences, self.subvector_num), dtype=torch.int32, device='cpu')
        for i in tqdm(range(start, ending, batch_size), desc=f'Document TopK Mapping {rank}'):
            batch_start = i
            batch_ending = min(i+batch_size, ending)
            cur_embeddings = torch.tensor(
                doc_embeddings[batch_start:batch_ending], device='cpu')
            all_topk_label[batch_start-start:batch_ending-start] = self.beam_search(
                cur_embeddings, num_return_sequences)
        return all_topk_label

    def get_reconstruct_loss_for_embeddings(self, embeddings, labels):
        batch_size = labels.shape[0]
        nidim = len(labels.shape)
        assert nidim == 2
        codebook = self.get_codebook()[..., :self.last_dim].unsqueeze(
            0).expand(labels.size(0), -1, -1, -1)
        expand_target = (-1,) * (nidim + 1) + (self.last_dim,)
        labels = labels.unsqueeze(-1).unsqueeze(-1).expand(*expand_target)
        vectors = codebook.gather(-2, index=labels).squeeze(-2)
        if self.pq_type == 'pq':
            diff = embeddings - vectors.view(batch_size, -1)
        elif self.pq_type == 'rq':
            diffs = []
            cur_embeddings = embeddings
            for i in range(self.subvector_num):
                cur_embeddings = cur_embeddings - vectors[:, i, :]
                diffs.append(cur_embeddings)
            diff = torch.stack(diffs, 1)
        else:
            assert False
        loss = (diff ** 2).mean()
        # current using mean on all
        # TODO: try sum in sample and mean cross samples
        return loss

    def get_reconstruct_vector(self, index, codebook=None):
        nidim = len(index.shape)
        assert nidim in (1, 2)
        if codebook is None:
            # assert nidim == 1
            codebook = self.get_codebook()[..., :self.last_dim]
        expand_target = (-1,) * (nidim + 1) + (self.last_dim,)
        vectors = codebook.gather(
            -2, index=index.unsqueeze(-1).unsqueeze(-1).expand(*expand_target)).squeeze(-2)
        if self.pq_type == 'rq':
            vectors = torch.sum(vectors, dim=-2)
        else:
            view_target = (index.shape[0], -1) if nidim == 2 else (-1,)
            vectors = vectors.view(view_target)
        if self.pq_type == 'opq':
            vectors = torch.matmul(vectors, self.rotate)
        return vectors

    def get_reconstruct_vector_matrix_multiply(self, index):
        # index: (bs, nsub, ncent); codebook: (nsub, ncent, last_dim)
        bs = index.shape[0]
        codebook = self.get_codebook()[..., :self.last_dim]
        index = index.view(-1, self.subvector_cents).unsqueeze(1)
        codebook = codebook.unsqueeze(0).expand(
            bs, -1, -1, -1).reshape(-1, self.subvector_cents, self.last_dim)
        output = torch.bmm(index, codebook).squeeze(
            1).view(bs, self.subvector_num, self.last_dim)
        if self.pq_type == 'rq':
            output = torch.sum(output, dim=1)
        else:
            output = output.view(bs, -1)
        return output


if __name__ == '__main__':
    # offline generate faiss index
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dist_mode', type=str, default='l2')
    parser.add_argument('--pq_type', type=str, default='pq')
    parser.add_argument('--subvector_num', type=int, default=4)
    parser.add_argument('--subvector_bits', type=int, default=4)
    parser.add_argument('--dim', type=int, default=768)
    args = parser.parse_args()
    assert args.dist_mode in ('l2', 'ip', 'iptol2')
    assert args.pq_type in ('pq', 'opq', 'rq')

    doc_embeddings = np.memmap(
        args.embedding_path, dtype=np.float32, mode='r').reshape(-1, args.dim)

    pq = ProductQuantization(
        args.pq_type,
        args.subvector_num,
        args.subvector_bits,
        args.dist_mode,
        args.emb_size,
    )
    pq.build_faiss_index(doc_embeddings, args.save_path)
