import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. FFT -> linear transform on low modes -> inverse FFT.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of kept frequency modes

        scale = 1.0 / (in_channels * out_channels)
        # Complex weights as (real, imag)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def compl_mul1d(self, a, b):
        """
        a: (B, Cin, M, 2)
        b: (Cin, Cout, M, 2)
        returns: (B, Cout, M, 2)
        """
        ar, ai = a[..., 0], a[..., 1]  # (B, Cin, M)
        br, bi = b[..., 0], b[..., 1]  # (Cin, Cout, M)

        real = torch.einsum("bcm,com->bom", ar, br) - torch.einsum("bcm,com->bom", ai, bi)
        imag = torch.einsum("bcm,com->bom", ar, bi) + torch.einsum("bcm,com->bom", ai, br)
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """
        x: (B, C_in, N)
        returns: (B, C_out, N)
        """
        B, C, N = x.shape

        # rFFT over last dim
        x_ft = torch.fft.rfft(x, dim=-1, norm="ortho")  # (B, C, N//2+1), complex
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (B, C, N//2+1, 2)

        Mmax = x_ft.shape[-2]  # N//2 + 1
        M = min(self.modes, Mmax)

        # Allocate output in Fourier domain
        out_ft = torch.zeros(B, self.out_channels, Mmax, 2, device=x.device, dtype=x.dtype)

        # Low-frequency block
        out_ft[:, :, :M] = self.compl_mul1d(x_ft[:, :, :M], self.weights[:, :, :M])

        # Back to complex then inverse rFFT
        out_ft_c = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft(out_ft_c, n=N, dim=-1, norm="ortho")  # (B, Cout, N)
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator with 3 Fourier layers.
    Input:  (B, 2, 201)
    Output: (B, 1, 201)
    """
    def __init__(self, modes=32, width=64, in_channels=2, out_channels=1):
        super().__init__()
        self.width = width
        self.ACT = F.relu

        # First projection to width channels
        self.fc0 = nn.Conv1d(in_channels, width, kernel_size=1)

        # 3 Fourier layers with 1x1 conv skips
        self.s1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, kernel_size=1)

        self.s2 = SpectralConv1d(width, width, modes)
        self.w2 = nn.Conv1d(width, width, kernel_size=1)

        self.s3 = SpectralConv1d(width, width, modes)
        self.w3 = nn.Conv1d(width, width, kernel_size=1)

        # Projection to single-channel output
        self.proj1 = nn.Conv1d(width, 256, kernel_size=1)
        self.proj2 = nn.Conv1d(256, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C_in, N)
        returns: (B, 1, N)
        """
        x = self.fc0(x)

        x1 = self.s1(x)
        x = self.ACT(x1 + self.w1(x))

        x1 = self.s2(x)
        x = self.ACT(x1 + self.w2(x))

        x1 = self.s3(x)
        x = x1 + self.w3(x)

        x = self.ACT(self.proj1(x))
        x = self.proj2(x)
        return x


# ---------------- Test code ----------------
if __name__ == "__main__":
    B, Cin, N = 4, 2, 201
    x = torch.randn(B, Cin, N)

    model = FNO1d(modes=64, width=64, in_channels=2, out_channels=1)
    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)  # (B,2,201)
    print("Output shape:", y.shape)  # (B,1,201)
