import torch

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


if __name__ == '__main__':
    # test TriangularCausalMask
    B, L = 2, 5
    mask = TriangularCausalMask(B, L)
    print(mask.mask)
    print(mask.mask.shape)

    # # test ProbMask
    # B, H, L = 2, 2, 5
    # index = torch.tensor([[0, 1, 2, 3, 4],
    #                       [0, 1, 2, 3, 4]])
    # scores = torch.randn(B, H, L, L)
    # mask = ProbMask(B, H, L, index, scores)
    # print(mask.mask)
    # print(mask.mask.shape)

