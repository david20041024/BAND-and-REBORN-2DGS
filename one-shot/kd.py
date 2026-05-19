import torch

class KD:
    def __init__(self, xyz: torch.Tensor, mesh: torch.Tensor, normal: torch.Tensor,alpha: torch.Tensor, max_size: int = 512):
        self.max_size = max_size
        self.device = xyz.device
        self.total_N = xyz.size(0) # 記住全局 N 大小
        
        self.groups = []
        self.group_xyz_mean = []  
        self.group_normal_mean = []  
        self.sorted_groups = []
        self.alpha = alpha
        self.counts = torch.zeros(len(self.groups), dtype=torch.int32, device=self.device)
        self.clone_count = torch.zeros(self.total_N, dtype=torch.int32, device=self.device)
        # 1. 拼接資料 (N, 7)
        data = torch.cat([xyz, normal], dim=1)  
        
        # 2. 建樹
        self.root = self._build(data, depth=0)
        
        # 3. 初始化計數陣列
        
        
        # 4. 高效批次並行查詢 Mesh，更新 self.counts
        self._query_all_parallel(mesh)
        
        # 5. 根據curvature排序每個group內的點
        self._count_curvature()

        # 6. 計算每個要複製的數量
        self._count_clone()


    def _build(self, points: torch.Tensor, depth: int) -> dict:
        N = points.size(0)
        if N <= self.max_size:
            node = {'leaf': True, 'group_idx': len(self.groups)}
            self.groups.append(points)
            self.group_xyz_mean.append(points[:, :4].mean(dim=0))   
            self.group_normal_mean.append(points[:, 4:].mean(dim=0)) 
            return node

        axis = depth % 3  
        idx = torch.argsort(points[:, axis])
        sorted_points = points[idx]
        mid = N // 2

        # 這裡加入一個防死循環的安全鎖
        if mid == 0 or mid == N:
            node = {'leaf': True, 'group_idx': len(self.groups)}
            self.groups.append(points)
            self.group_xyz_mean.append(points[:, :4].mean(dim=0))   
            self.group_normal_mean.append(points[:, 4:].mean(dim=0)) 
            return node

        return {
            'leaf': False,
            'axis': axis,
            'split_val': sorted_points[mid, axis].item(),
            'left':  self._build(sorted_points[:mid], depth + 1),
            'right': self._build(sorted_points[mid:], depth + 1),
        }

    def _query_all_parallel(self, mesh: torch.Tensor):
        # 丟棄傳統 single-point 迴圈，改用大批次 Tensor Mask 遞迴投遞
        self._dispatch_mesh_batch(self.root, mesh)

    def _dispatch_mesh_batch(self, node: dict, mesh_subset: torch.Tensor):
        if mesh_subset.size(0) == 0:
            return
        if node['leaf']:
            self.counts[node['group_idx']] += mesh_subset.size(0)
            return
            
        axis = node['axis']
        split_val = node['split_val']
        
        left_mask = mesh_subset[:, axis] < split_val
        self._dispatch_mesh_batch(node['left'], mesh_subset[left_mask])
        self._dispatch_mesh_batch(node['right'], mesh_subset[~left_mask])

    def _count_curvature(self):

        for g_idx, points in enumerate(self.groups):
            if points.size(0) == 0:
                self.sorted_groups.append(points)
                continue
                
            orig_idx = points[:, 3].long()  # 拿出第四維的原始身分證
            xyz = points[:, :3]
            normal = points[:, 4:]
            

            normal_mean = self.group_normal_mean[g_idx]
            diff_normal = normal - normal_mean.unsqueeze(0)
            curv_val = torch.norm(diff_normal, dim=1, keepdim=True) # 得到 (Ka, 1) 的曲率
            
            final_score = curv_val * self.alpha[orig_idx]

            _, sort_idx = torch.sort(final_score, dim=0, descending=True)

            sorted_points = points[sort_idx.squeeze(1)]
            
            # 5. 存入結果
            self.sorted_groups.append(sorted_points)
    
     def _count_clone(self):
        
        for g_idx, sorted_points in enumerate(self.sorted_groups):
            B = sorted_points.size(0)  # 這組總共有多少個高斯點
            if B == 0:
                continue
                
            C = self.counts[g_idx].item()  # 這組拿到的總牌數 C
            if C == 0:
                continue
                
            orig_idx = sorted_points[:, 3].long()  # 原始身分證
            
            # 1. 先建一個大小為 C 的序列，內容是 [0, 1, 2, 0, 1, 2, 0...] 代表牌要發給誰
            # 透過對 B 取餘數 (%)，就能無限循環這組點的局部 Index
            card_indices = torch.arange(C, device=self.device) % B
            
            # 2. 用 torch.bincount 一口氣計算每個人（0 到 B-1）總共被分配到幾張牌！
            # minlength=B 可以確保哪怕後面的人沒分到牌（C < B），矩陣長度依然是 B
            local_clone = torch.bincount(card_indices, minlength=B).to(torch.int32)
            
            # 3. 精準填回全局
            self.clone_count[orig_idx] = local_clone

        