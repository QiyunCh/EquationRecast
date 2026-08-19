import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier layer (FNO-style):
      - rFFT2
      - learnable linear transform on low-frequency modes
      - include both ky blocks: [0..m-1] and [-m..-1]
      - inverse rFFT2

    Input/Output:
      x: (B, Cin, Ny, Nx)
      y: (B, Cout, Ny, Nx)

    Notes:
      - Using both positive and negative ky low-mode blocks is critical.
        Using only the first ky block often produces directional artifacts (e.g., vertical stripes).
    """
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)

        # Scale for stable init
        scale = 1.0 / (in_channels * out_channels)

        # Two sets of complex weights:
        #   - weights_pos for ky in [0 .. modes_y-1]
        #   - weights_neg for ky in [-modes_y .. -1]
        #
        # Shape convention:
        #   (Cin, Cout, My, Mx, 2) where last dim is (real, imag)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes_y, self.modes_x, 2)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.modes_y, self.modes_x, 2)
        )

    @staticmethod
    def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication with summation over Cin.

        a: (B, Cin, My, Mx, 2)
        b: (Cin, Cout, My, Mx, 2)
        out: (B, Cout, My, Mx, 2)
        """
        ar, ai = a[..., 0], a[..., 1]
        br, bi = b[..., 0], b[..., 1]
        real = torch.einsum("bcij,coij->boij", ar, br) - torch.einsum("bcij,coij->boij", ai, bi)
        imag = torch.einsum("bcij,coij->boij", ar, bi) + torch.einsum("bcij,coij->boij", ai, br)
        return torch.stack([real, imag], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, Ny, Nx)
        returns: (B, Cout, Ny, Nx)
        """
        B, Cin, Ny, Nx = x.shape
        assert Cin == self.in_channels, f"Expected Cin={self.in_channels}, got {Cin}"

        # rFFT2 => (B, Cin, Ny, Nx//2+1) complex
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (B, Cin, Ny, Nx//2+1, 2)

        # Available mode limits from rFFT2
        My_max = x_ft.shape[-3]          # Ny
        Mx_max = x_ft.shape[-2]          # Nx//2 + 1
        My = min(self.modes_y, My_max)
        Mx = min(self.modes_x, Mx_max)

        # Output spectrum
        out_ft = torch.zeros(
            B, self.out_channels, My_max, Mx_max, 2,
            device=x.device, dtype=x_ft.dtype
        )

        # Positive ky block: ky = 0..My-1, kx = 0..Mx-1
        out_ft[:, :, :My, :Mx] = self._compl_mul2d(
            x_ft[:, :, :My, :Mx],
            self.weights_pos[:, :, :My, :Mx]
        )

        # Negative ky block: ky = -My..-1, kx = 0..Mx-1
        # This is critical for isotropy / avoiding directional artifacts.
        out_ft[:, :, -My:, :Mx] = self._compl_mul2d(
            x_ft[:, :, -My:, :Mx],
            self.weights_neg[:, :, :My, :Mx]
        )

        # Back to complex and inverse rFFT2
        out_ft_c = torch.complex(out_ft[..., 0], out_ft[..., 1])
        y = torch.fft.irfft2(out_ft_c, s=(Ny, Nx), dim=(-2, -1), norm="ortho")
        return y


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Default recommendation for your setting (Ny=Nx=128, main energy k~20):
      - modes_x = modes_y = 32 (or 24 for more aggressive low-pass)
      - width ~ 64

    Input:
      x: (B, in_channels, Ny, Nx)  (you currently use in_channels=1 => source S)
    Output:
      y: (B, out_channels, Ny, Nx) (out_channels=1 => omega)
    """
    def __init__(
        self,
        modes_x: int = 32,
        modes_y: int = 32,
        width: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
        use_gelu: bool = True,
    ):
        super().__init__()
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.act = F.gelu if use_gelu else F.relu

        # 1x1 "lifting" (no padding => safe for periodic)
        self.fc0 = nn.Conv2d(in_channels, self.width, kernel_size=1, padding=0, bias=True)

        self.spectral_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv2d(self.width, self.width, modes_x, modes_y))
            # 1x1 skip conv (no padding => safe for periodic)
            self.skip_layers.append(nn.Conv2d(self.width, self.width, kernel_size=1, padding=0, bias=True))

        # Projection
        self.proj1 = nn.Conv2d(self.width, 256, kernel_size=1, padding=0, bias=True)
        self.proj2 = nn.Conv2d(256, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)

        # Stacked Fourier layers
        for i in range(self.n_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.skip_layers[i](x)
            x = self.act(x1 + x2)

        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x


if __name__ == "__main__":
    # Quick sanity test
    B, Cin, Ny, Nx = 2, 1, 128, 128
    x = torch.randn(B, Cin, Ny, Nx)
    model = FNO2d(modes_x=32, modes_y=32, width=64, in_channels=1, out_channels=1)
    with torch.no_grad():
        y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
