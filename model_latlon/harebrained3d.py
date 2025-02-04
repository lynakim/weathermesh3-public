from model_latlon.harebrained2d import *
from model_latlon.primatives3d import *

DEFAULT_KERNEL_DEPTH = 1

def get_full_dims3d(strips):
    H = sum([strip.shape[3] for strip in strips])
    B,C,D,_,W = get_center(strips).shape
    return B,C,D,H,W 

class ToHarebrained3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_depth=DEFAULT_KERNEL_DEPTH, kernel_height=DEFAULT_KERNEL_HEIGHT, strip_edges=EDGES_DEFAULT, strip_strides=STRIDES_DEFAULT):
        super(ToHarebrained3d, self).__init__()
        assert strip_edges[-1] == 1.0
        assert len(strip_edges) == len(strip_strides)
        self.strides = strip_strides  
        self.kernel_widths = [(kernel_height//2*2)*x+1 for x in self.strides]
        self.conv_center = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_depth, kernel_height, self.kernel_widths[0]))
        self.strip_edges = strip_edges
        self.kernel_depth = kernel_depth
        self.kernel_height = kernel_height
        self.convs = nn.ModuleList()
        for i in range(1, len(strip_edges)):
            conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(kernel_depth, kernel_height, self.kernel_widths[i]),
                stride=(1, 1, strip_strides[i])
            )
            self.convs.append(conv)

    def forward(self, x):
        B, C, D, H, W = x.shape
        for s in self.strides:
            assert W % s == 0, "Width must be divisible by all strides."
        assert (H - 1) % 2 == 0, f"Height must be odd, got {H}"
        hpad = self.kernel_height // 2
        dpad = self.kernel_depth // 2

        # Pad the input tensor
        x_pad = earth_pad3d(x, (dpad, hpad, 0))  # Pad depth and height

        def f2idx(f, dir):
            # f is a float from 0.0 (center) to 1.0 (edge)
            # dir specifies direction from center: -1 (north), 1 (south)
            idx = H // 2 + hpad + int(dir == 1) + dir * int(H // 2 * f)
            return idx

        # Process the central strip
        ctp = f2idx(self.strip_edges[0], -1) - hpad
        cbp = f2idx(self.strip_edges[0], 1) + hpad
        center = x_pad[:, :, :, ctp:cbp, :]
        center_pad = earth_pad3d(center, (0, 0, self.kernel_widths[0] // 2))
        center_out = self.conv_center(center_pad)

        hemispheres = []
        for dir in [-1, 1]:  # -1: north, 1: south
            strips = []
            indices = sorted(range(1, len(self.strip_edges)), reverse=(dir == -1))
            for i in indices:
                sl = sorted([
                    f2idx(self.strip_edges[i], dir),
                    f2idx(self.strip_edges[i - 1], dir)
                ])
                sl = [sl[0] - hpad, sl[1] + hpad]  # Adjust for padding
                strip = x_pad[:, :, :, sl[0]:sl[1], :]
                strip_pad = earth_pad3d(strip, (0, 0, self.kernel_widths[i] // 2))
                strip_out = self.convs[i - 1](strip_pad)
                strips.append(strip_out)
            hemispheres.append(strips)

        # Combine outputs
        out = hemispheres[0] + [center_out] + hemispheres[1]  # [north strips, center, south strips]
        nB,nC,nD,nH,nW = get_full_dims3d(out)
        assert (B,D,H,W) == (nB,nD,nH,nW), f"Rip, something is bad, {nB},{nD},{nH},{nW} vs {B},{D},{H},{W}"
        return out

class FromHarebrained3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_depth=DEFAULT_KERNEL_DEPTH, kernel_height=DEFAULT_KERNEL_HEIGHT, strip_edges=EDGES_DEFAULT, strip_strides=STRIDES_DEFAULT):
        super(FromHarebrained3d, self).__init__()
        assert strip_edges[-1] == 1.0
        assert kernel_height % 2 == 1
        assert kernel_height >= 3
        assert len(strip_edges) == len(strip_strides)
        self.strides = strip_strides  
        self.kernel_widths = [(kernel_height // 2 * 2) * x + 1 for x in self.strides]

        # Biases can interfere with wrapping on a conv up. If you want a bias, add it manually after the wrap.
        self.conv_center = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(kernel_depth, kernel_height, self.kernel_widths[0]),
            bias=False
        )

        self.strip_edges = strip_edges
        self.kernel_height = kernel_height
        self.kernel_depth = kernel_depth
        self.convs = nn.ModuleList()
        for i in range(1, len(strip_edges)):
            up = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=(kernel_depth, kernel_height, self.kernel_widths[i]),
                stride=(1, 1, strip_strides[i]),
                bias=False
            )
            self.convs.append(up)

    def forward(self, strips):
        N = (len(self.strides) - 1) * 2 + 1
        assert len(strips) == N, f"strip length mismatch, {len(strips)} vs {N}"
        center = get_center(strips)
        B, C, D, H, W = get_full_dims3d(strips)
        Cout = self.conv_center.out_channels
        hpad = self.kernel_height // 2
        dpad = self.kernel_depth // 2
        wpadmax = max(self.kernel_widths) // 2
        Hpad = H + hpad * 2
        Dpad = D + dpad * 2
        Wpad = W + wpadmax * 2

        def f2idx(f, dir):
            # f is a float from 0.0 (center) to 1.0 (edge)
            # dir specifies direction from center: -1 (north), 1 (south)
            idx = H // 2 + hpad + int(dir == 1) + dir * int(H // 2 * f)
            return idx

        # Initialize padded output tensor
        x_pad = center.new_zeros(B, Cout, Dpad, Hpad, Wpad)
        wpadc = self.kernel_widths[0] // 2
        dpadc = self.kernel_depth // 2

        # Process center strip
        c_out = self.conv_center(center)
        ctp = f2idx(self.strip_edges[0], -1) - hpad
        cbp = f2idx(self.strip_edges[0], 1) + hpad
        x_pad[:, :, dpadc:Dpad-dpadc, ctp:cbp, wpadmax - wpadc : Wpad - (wpadmax - wpadc)] += c_out

        # Process outer strips
        ostrips = strips[:N // 2] + strips[N // 2 + 1:]
        idxs = list(reversed(range(1, N // 2 + 1))) + list(range(1, N // 2 + 1))
        dirs = [-1] * (N // 2) + [1] * (N // 2)

        for strip, i, dir in zip(ostrips, idxs, dirs):
            sl = sorted([
                f2idx(self.strip_edges[i], dir),
                f2idx(self.strip_edges[i - 1], dir)
            ])
            sl = [sl[0] - hpad, sl[1] + hpad]
            out = self.convs[i - 1](strip)
            wpad = self.kernel_widths[i] // 2
            iL = wpadmax - wpad
            iR = iL + out.shape[-1]
            x_pad[:, :, dpadc:Dpad-dpadc, sl[0]:sl[1], iL:iR] += out

        # Wrap around in the longitude (W) dimension
        x_pad[:, :, :, :, wpadmax : 2 * wpadmax] += x_pad[:, :, :, :, W + wpadmax :]
        x_pad[:, :, :, :, -2 * wpadmax : -wpadmax] += x_pad[:, :, :, :, : -(W + wpadmax)]

        # Trim padding to get the final output
        x = x_pad[:, :, dpadc:Dpad-dpadc, hpad:-hpad, wpadmax:-wpadmax]

        return x

class HarebrainedPad3d(nn.Module):
    def __init__(self, channels, kernel_height=DEFAULT_KERNEL_HEIGHT, strip_strides=STRIDES_DEFAULT):
        super(HarebrainedPad3d, self).__init__()
        self.strides = strip_strides
        self.kernel_height = kernel_height
        self.to_outers = nn.ModuleList()
        self.to_inners = nn.ModuleList()
        for i in range(1, len(strip_strides)):
            assert strip_strides[i] % strip_strides[i - 1] == 0
            stride = strip_strides[i] // strip_strides[i - 1]
            to_outer = nn.Conv3d(
                channels, channels,
                kernel_size=(1, 1, 2 * stride + 1),
                stride=(1, 1, stride)
            )
            to_inner = nn.ConvTranspose3d(
                channels, channels,
                kernel_size=(1, 1, 2 * stride + 1),
                stride=(1, 1, stride),
                bias=False
            )
            self.to_inners.append(to_inner)
            self.to_outers.append(to_outer)

    def forward(self, strips):
        N = (len(self.strides) - 1) * 2 + 1
        assert len(strips) == N, f"strip length mismatch, {len(strips)} vs {N}"

        def get_edge_slices(dir, pad_h):
            if dir == 1:
                # Positive dir gets bottom edge of tensor in height dimension
                return (slice(None), slice(None), slice(None), slice(-pad_h, None), slice(None))
            elif dir == -1:
                # Negative dir gets top edge of tensor in height dimension
                return (slice(None), slice(None), slice(None), slice(None, pad_h), slice(None))
            else:
                assert False

        pad = self.kernel_height // 2
        strips_pad = [earth_pad3d(x, (0, pad, pad)) for x in strips]

        c_pad = get_center(strips_pad)
        for dir in [-1, 1]:
            a = get_strip(strips, dir, 1)[get_edge_slices(-dir, pad)]  # Invert dir to get the opposite edge
            a = self.to_inners[0](a)
            a = earth_wrap3d(a, get_center(strips).shape[-1], self.to_inners[0].kernel_size[-1] // 2)
            a = earth_pad3d(a, (0, 0, pad))
            c_pad[get_edge_slices(dir, pad)] = a

        def pad_strip(dir, mi):
            s_pad = get_strip(strips_pad, dir, mi)
            if mi + 1 > N // 2:
                pass  # At the poles, no outer strips
            else:
                # Pad from the outer strip
                o = get_strip(strips, dir, mi + 1)[get_edge_slices(-dir, pad)]
                o = self.to_inners[mi](o)
                o = earth_wrap3d(o, get_strip(strips, dir, mi).shape[-1], self.to_inners[mi].kernel_size[-1] // 2)
                o = earth_pad3d(o, (0, 0, pad))
                s_pad[get_edge_slices(dir, pad)] = o

            # Pad from the inner strip
            en = get_strip(strips, dir, mi - 1)[get_edge_slices(dir, pad)]
            en = earth_pad3d(en, (0, 0, self.to_outers[mi - 1].kernel_size[-1] // 2))
            en = self.to_outers[mi - 1](en)
            en = earth_pad3d(en, (0, 0, pad))
            s_pad[get_edge_slices(-dir, pad)] = en

        for i in range(N // 2):
            pad_strip(-1, i + 1)
            pad_strip(1, i + 1)

        return strips_pad