import torch
import torch.nn.functional as F


def feat2token(feat, block_size=(4, 4), stride=None, padding=None):
    """
       Tokenize a 4D feature map into 3D sequences.
       Each sequence is treated as a new batch containing multiple tokens. 
       `feat` -> torch.Tensor
    """
    assert isinstance(feat, torch.Tensor), ValueError
    assert len(feat.size()) == 4, ValueError

    if stride is None:
        stride = block_size

    b, c, h, w = feat.size()
    assert (not h % stride[0]) and (not w % stride[1]), ValueError
    
    num_tokens = (h // stride[0], w // stride[1])
    if padding is None:
        padding = [0, 0, 0, 0]
    else:
        padding = padding
    feat = F.pad(feat, padding, 'constant', 0)

    tokens = []
    for _h in range(num_tokens[0]):
        for _w in range(num_tokens[1]):
            # (b, c, block_size[0], block_size[1])
            _block= feat[:, :, _h*stride[0]: _h*stride[0]+block_size[0], _w*stride[1]: _w*stride[1]+block_size[1]]

            # Razer-scan each block into a token: (b, block_size[0]*block_size[1], c)
            _token = _block.permute(0, 2, 3, 1).contiguous().view(b, block_size[0]*block_size[1], c)

            tokens.append(_token)

    tokens = torch.cat(tokens, dim=0)

    return tokens
 

def token2feat(tokens, block_size=(4, 4), feat_size=(16, 3, 16, 16)):
    """
       Convert tokenized 3D sequences into 4D feature map.
       When converting with `feat2token()`, it requires
            * block_size==stride
            * padding == 0
    """
    b, t, c = tokens.size()
    feat = torch.zeros(feat_size).to(tokens.device)

    assert t == block_size[0]*block_size[1], ValueError # Sequence length
    assert (not b%feat_size[0]) and (c == feat_size[1]), ValueError

    num_tokens = (feat_size[2] // block_size[0], feat_size[3] // block_size[1])
    tokens = tokens.chunk(b//feat_size[0])

    for _h in range(num_tokens[0]):
        for _w in range(num_tokens[1]):
            _block = tokens[_h*num_tokens[1]+_w].view(feat_size[0], block_size[0], block_size[1], c).contiguous()

            feat[:, :, _h*block_size[0]:(_h+1)*block_size[0], _w*block_size[1]:(_w+1)*block_size[1]] = _block.permute(0, 3, 1, 2)

    return feat


if __name__ == '__main__':
    a = torch.arange(2*3*4*8).view(2, 3, 4, 8)
    print('a ', a, a.shape)
    tokens = feat2token(a, (2, 2), (2, 2))
    print('tokens ', tokens, tokens.shape)
    feat = token2feat(tokens, (2, 2), (2, 3, 4, 8))
    print('feat ', feat, feat.shape)
    #overlap_tokens = feat2token(a, (4, 4), (2, 2), [1]*4)
    #print('overlap_tokens ', overlap_tokens, overlap_tokens.shape)
