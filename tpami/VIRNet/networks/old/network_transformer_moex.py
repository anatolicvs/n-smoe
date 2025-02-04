# MoE_2
# if cfg.DecoderConfig.kernel_type == KernelType.GAUSSIAN:
#     d_out = (7 * cfg.d_in) * cfg.DecoderConfig.kernel
# else:
#     d_out = (7 * cfg.d_in + 3) * cfg.DecoderConfig.kernel


class MoE_(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.min_diag = 1e-6
        self.min_denominator = 1e-8

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)  # [W]
        yy = torch.linspace(0.0, 1.0, height, device=device)  # [H]
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")  # Each [W, H]
        grid = torch.stack((gx, gy), dim=-1)  # [W, H, 2]
        return grid.float()  # [W, H, 2]

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)  # [B, ch, k, 1]
        st = torch.sin(theta).unsqueeze(-1)  # [B, ch, k, 1]
        R = torch.cat([ct, -st, st, ct], dim=-1)  # [B, ch, k, 4]
        return R.view(*theta.shape, 2, 2)  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size1(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 1, 1, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 1, 1]
        L[..., 0, 0] = F.softplus(params[..., 0]) + 1e-2  # [B, ch, k]
        return L  # [B, ch, k, 1, 1]

    def construct_lower_triangular_size2(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 3]
        l11, l21, l22 = torch.split(params, 1, dim=-1)  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 2, 2, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 2, 2]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size3(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 6]
        l11, l21, l22, l31, l32, l33 = torch.split(
            params, 1, dim=-1
        )  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        l33 = F.softplus(l33) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 3, 3, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 3, 3]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        L[..., 2, 0] = l31.squeeze(-1)  # [B, ch, k]
        L[..., 2, 1] = l32.squeeze(-1)  # [B, ch, k]
        L[..., 2, 2] = l33.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 3, 3]

    def construct_lower_triangular(
        self, params: torch.Tensor, size: int
    ) -> torch.Tensor:
        if size == 1:
            return self.construct_lower_triangular_size1(params)  # [B, ch, k, 1, 1]
        elif size == 2:
            return self.construct_lower_triangular_size2(params)  # [B, ch, k, 2, 2]
        elif size == 3:
            return self.construct_lower_triangular_size3(params)  # [B, ch, k, 3, 3]
        else:
            raise ValueError(
                f"Unsupported size: {size}. Only size=1, 2, and 3 are supported."
            )

    def cov_mat(
        self,
        L_spatial: torch.Tensor,
        theta_xy: torch.Tensor,
        L_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(
            R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1))
        )  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))  # [B, ch, k, 2, 2]
        C_xy = 0.5 * (C_xy + C_xy.mT)  # [B, ch, k, 2, 2] Ensure symmetry
        if ch == 1:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,1,k,1,1]
            C_color = C_color.squeeze(-1).squeeze(-1)  # [B,1,k]
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,1,k,3,3]
            C_full[..., :2, :2] = C_xy  # [B,1,k,2,2]
            C_full[..., 2, 2] = C_color  # [B,1,k]
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,3,k,3,3]
            B_, ch_, k_ = C_xy.shape[:3]
            C_full = torch.zeros(
                B_, ch_, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,3,k,5,5]
            C_full[..., :2, :2] = C_xy  # [B,3,k,2,2]
            C_full[..., 2:, 2:] = C_color  # [B,3,k,3,3]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        return C_full * self.sharpening_factor  # [B, ch, k, D, D]

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape  # [B, ch, k * param_per_kernel]
        p = p.view(B, ch, k, -1)  # [B, ch, k, param_per_kernel]
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, size=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))  # [B, ch, k]
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 9:10].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 9:15].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, size=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, size=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 7:8].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 7:13].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, size=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, None, None
        else:
            raise NotImplementedError(
                f"Kernel type {self.kernel_type} not implemented."
            )

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, W, H, D]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)  # [B, ch, k, W, H]
        Sigma_inv_diag = Sigma_inv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        denominator = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv_diag.clamp(
            min=self.min_diag
        )  # [B, ch, k, 1, 1]
        denominator = denominator.clamp(min=self.min_denominator)
        C_csigma = 1.0 / (1.0 + norm_x**2 / denominator)  # [B, ch, k, W, H]
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )  # [B, ch, k, W, H]
        return combined  # [B, ch, k, W, H]

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        return G_sigma  # [B, ch, k, W, H]

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape  # [B, ch, k * param_per_kernel]
        k = self.kernel  # int
        mu_xy, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)
        # mu_xy: [B, ch, k, D]
        # cov_matrix: [B, ch, k, D, D]
        # w: [B, ch, k]
        # alpha: [B, ch, k] or None
        # c: [B, ch, k] or None

        d = cov_matrix.shape[-1]  # D=3 or 5
        eye_d = torch.eye(d, device=cov_matrix.device).view(
            1, 1, 1, d, d
        )  # [1,1,1,D,D]

        # cov_matrix_reg = cov_matrix + (self.min_diag + 1e-3) * eye_d  # [B, ch, k, D, D]

        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        min_eig = eigenvalues.min(dim=-1).values[..., None, None]
        eps_adaptive = F.softplus(-min_eig) + 1e-8

        cov_matrix_reg = cov_matrix + (1e-6 + eps_adaptive) * eye_d

        L = torch.linalg.cholesky(cov_matrix_reg)  # [B, ch, k, D, D]
        Sigma_inv = torch.cholesky_inverse(L)  # [B, ch, k, D, D]
        # Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.transpose(-2, -1))  # [B, ch, k, D, D]

        # Sigma_inv = torch.linalg.solve(cov_matrix + eps * eye_d, eye_d)

        g = self.grid(height, width, params.device)  # [W, H, 2]
        g_expanded = g.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,1,W,H,2]
        g_expanded = g_expanded.repeat(B, ch, k, 1, 1, 1)  # [B, ch, k, W, H, 2]

        mu_full = mu_xy.unsqueeze(3).unsqueeze(4)  # [B, ch, k, 1, 1, D]

        if ch == 1:
            color_zeros = (
                torch.zeros_like(mu_xy[..., -1:]).unsqueeze(3).unsqueeze(4)
            )  # [B,1,k,1,1,1]
            color_zeros = color_zeros.expand(
                -1, -1, -1, height, width, -1
            )  # [B,1,k,H,W,1]
            x = torch.cat([g_expanded, color_zeros], dim=-1)  # [B,1,k,W,H,3]
        elif ch == 3:
            color_mean = mu_xy[..., -3:].unsqueeze(3).unsqueeze(4)  # [B,3,k,1,1,3]
            color_mean_expanded = color_mean.expand(
                -1, -1, -1, height, width, -1
            )  # [B,3,k,H,W,3]
            x = torch.cat([g_expanded, color_mean_expanded], dim=-1)  # [B,3,k,W,H,5]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(
                x, mu_full, Sigma_inv, alpha, c
            )  # [B, ch, k, W, H]
        else:
            mu_spatial = mu_xy[..., :2].reshape(B, ch, k, 1, 1, 2)  # [B, ch, k,1,1,2]
            Sigma_inv_spatial = Sigma_inv[..., :2, :2]  # [B, ch, k,2,2]
            ker = self.gaussian_kernel(
                g_expanded, mu_spatial, Sigma_inv_spatial
            )  # [B, ch, k, W, H]

        # detJ = torch.det(Sigma_inv).sqrt()
        # ker = ker * detJ.unsqueeze(-1).unsqueeze(-1)

        ker = ker * w.view(B, ch, k, 1, 1)  # [B, ch, k, W, H]
        ker_sum = ker.sum(dim=2, keepdim=True)  # [B, ch, 1, W, H]
        ker = ker / (ker_sum + 1e-8)  # [B, ch, k, W, H]
        out = ker.sum(dim=2)  # [B, ch, W, H]
        return torch.clamp(out, min=0.0, max=1.0)  # [B, ch, W, H]

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)


class MoE_4(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel  # Number of experts
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.min_diag = 1e-6
        self.min_denominator = 1e-8

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)  # [W]
        yy = torch.linspace(0.0, 1.0, height, device=device)  # [H]
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")  # gx: [W, H], gy: [W, H]
        grid = torch.stack((gx, gy), dim=-1)  # [W, H, 2]
        return grid.float()  # [W, H, 2]

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)  # [B, ch, k, 1]
        st = torch.sin(theta).unsqueeze(-1)  # [B, ch, k, 1]
        R = torch.cat([ct, -st, st, ct], dim=-1)  # [B, ch, k, 4]
        return R.view(*theta.shape, 2, 2)  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size1(self, params: torch.Tensor) -> torch.Tensor:
        # params: [B, ch, k, 1]
        L = torch.zeros(
            params.size(0),
            params.size(1),
            params.size(2),
            1,
            1,
            device=params.device,
            dtype=params.dtype,
        )  # [B, ch, k, 1, 1]
        L[..., 0, 0] = F.softplus(params[..., 0]) + self.min_diag  # [B, ch, k]
        return L  # [B, ch, k, 1, 1]

    def construct_lower_triangular_size2(self, params: torch.Tensor) -> torch.Tensor:
        # params: [B, ch, k, 3]
        l11, l21, l22 = torch.split(params, 1, dim=-1)  # Each: [B, ch, k, 1]
        l11 = F.softplus(l11) + self.min_diag  # [B, ch, k, 1]
        l22 = F.softplus(l22) + self.min_diag  # [B, ch, k, 1]
        L = torch.zeros(
            params.size(0),
            params.size(1),
            params.size(2),
            2,
            2,
            device=params.device,
            dtype=params.dtype,
        )  # [B, ch, k, 2, 2]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size3(self, params: torch.Tensor) -> torch.Tensor:
        # params: [B, ch, k, 6]
        l11, l21, l22, l31, l32, l33 = torch.split(
            params, 1, dim=-1
        )  # Each: [B, ch, k, 1]
        l11 = F.softplus(l11) + self.min_diag  # [B, ch, k, 1]
        l22 = F.softplus(l22) + self.min_diag  # [B, ch, k, 1]
        l33 = F.softplus(l33) + self.min_diag  # [B, ch, k, 1]
        L = torch.zeros(
            params.size(0),
            params.size(1),
            params.size(2),
            3,
            3,
            device=params.device,
            dtype=params.dtype,
        )  # [B, ch, k, 3, 3]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        L[..., 2, 0] = l31.squeeze(-1)  # [B, ch, k]
        L[..., 2, 1] = l32.squeeze(-1)  # [B, ch, k]
        L[..., 2, 2] = l33.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 3, 3]

    def construct_lower_triangular(
        self, params: torch.Tensor, size: int
    ) -> torch.Tensor:
        if size == 1:
            return self.construct_lower_triangular_size1(params)  # [B, ch, k, 1, 1]
        elif size == 2:
            return self.construct_lower_triangular_size2(params)  # [B, ch, k, 2, 2]
        elif size == 3:
            return self.construct_lower_triangular_size3(params)  # [B, ch, k, 3, 3]
        else:
            raise ValueError(
                f"Unsupported size: {size}. Only size=1, 2, and 3 are supported."
            )

    def cov_mat(
        self,
        L_spatial: torch.Tensor,  # [B, ch, k, 2, 2]
        theta_xy: torch.Tensor,  # [B, ch, k]
        L_color: torch.Tensor,  # [B, ch, k, size, size]
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(
            R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1))
        )  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))  # [B, ch, k, 2, 2]
        if ch == 1:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,1,k,1,1]
            B_, ch_, k_, _, _ = C_xy.shape
            C_full = torch.zeros(
                B_, ch_, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,1,k,3,3]
            C_full[..., :2, :2] = C_xy  # [B,1,k,2,2]
            C_full[..., 2, 2] = C_color.squeeze(-1)  # [B,1,k]
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,3,k,3,3]
            B_, ch_, k_, _, _ = C_xy.shape
            C_full = torch.zeros(
                B_, ch_, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,3,k,5,5]
            C_full[..., :2, :2] = C_xy  # [B,3,k,2,2]
            C_full[..., 2:, 2:] = C_color  # [B,3,k,3,3]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        return C_full * self.sharpening_factor  # [B, ch, k, D, D]

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # p: [B, ch, k * param_per_kernel]
        B, _, _ = p.shape

        p = p.view(B, ch, k, -1)  # [B, ch, k, param_per_kernel]
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            if ch == 1:
                mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
                L_spatial = self.construct_lower_triangular(
                    L_spatial_params, size=2
                )  # [B, ch, k, 2, 2]
                theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                    2 * torch.pi
                ) - torch.pi  # [B, ch, k]
                w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
                alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))  # [B, ch, k]
                c = (
                    F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag
                )  # [B, ch, k]
                L_color_params = torch.zeros_like(mu_x)  # [B, ch, k, 1]
                color_mean = torch.zeros_like(mu_x)  # [B, ch, k, 1]
            elif ch == 3:
                mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
                L_spatial = self.construct_lower_triangular(
                    L_spatial_params, size=2
                )  # [B, ch, k, 2, 2]
                theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                    2 * torch.pi
                ) - torch.pi  # [B, ch, k]
                w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
                alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))  # [B, ch, k]
                c = (
                    F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag
                )  # [B, ch, k]
                L_color_params = p[..., 9:15].reshape(B, ch, k, 6)  # [B, ch, k, 6]
                L_color = self.construct_lower_triangular(
                    L_color_params, size=3
                )  # [B, ch, k, 3, 3]
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            mu_xy = torch.cat(
                [mu_x, mu_y, color_mean], dim=-1
            )  # [B, ch, k, D=3 or D=5]
            if ch == 3:
                # L_color already defined
                pass
            elif ch == 1:
                L_color = self.construct_lower_triangular(
                    L_color_params, size=1
                )  # [B, ch, k, 1, 1]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, alpha, c, color_mean
        elif self.kernel_type == KernelType.GAUSSIAN:
            if ch == 1:
                mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
                L_spatial = self.construct_lower_triangular(
                    L_spatial_params, size=2
                )  # [B, ch, k, 2, 2]
                theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                    2 * torch.pi
                ) - torch.pi  # [B, ch, k]
                w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
                alpha = None
                c = None
                L_color_params = torch.zeros_like(mu_x)  # [B, ch, k, 1]
                color_mean = torch.zeros_like(mu_x)  # [B, ch, k, 1]
            elif ch == 3:
                mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
                L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
                L_spatial = self.construct_lower_triangular(
                    L_spatial_params, size=2
                )  # [B, ch, k, 2, 2]
                theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                    2 * torch.pi
                ) - torch.pi  # [B, ch, k]
                w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
                alpha = None
                c = None
                L_color_params = p[..., 7:13].reshape(B, ch, k, 6)  # [B,3,k,6]
                L_color = self.construct_lower_triangular(
                    L_color_params, size=3
                )  # [B, ch, k, 3, 3]
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            mu_xy = torch.cat(
                [mu_x, mu_y, color_mean], dim=-1
            )  # [B, ch, k, D=3 or D=5]
            if ch == 3:
                pass
            elif ch == 1:
                L_color = self.construct_lower_triangular(
                    L_color_params, size=1
                )  # [B, ch, k, 1, 1]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, alpha, c, color_mean
        else:
            raise NotImplementedError(
                f"Kernel type {self.kernel_type} not implemented."
            )

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,  # [B, ch, k, W, H, D=3 or D=5]
        mu: torch.Tensor,  # [B, ch, k, 1, 1, D]
        Sigma_inv: torch.Tensor,  # [B, ch, k, D, D]
        alpha: Optional[torch.Tensor],  # [B, ch, k]
        c: Optional[torch.Tensor],  # [B, ch, k]
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, W, H, D]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)  # [B, ch, k, W, H]

        Sigma_inv_diag = Sigma_inv[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        denominator = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv_diag.clamp(
            min=self.min_diag
        )  # [B, ch, k, 1, 1]
        denominator = denominator.clamp(min=self.min_denominator)  # [B, ch, k, 1, 1]
        C_csigma = 1.0 / (1.0 + norm_x**2 / denominator)  # [B, ch, k, W, H]
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )  # [B, ch, k, W, H]
        return combined  # [B, ch, k, W, H]

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, ch, k, W, H, 2]
        # mu_spatial: [B, ch, k, 1, 1, 2]
        # Sigma_inv_spatial: [B, ch, k, 2, 2]
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        return G_sigma  # [B, ch, k, W, H]

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        # params: [B, ch, k * param_per_kernel]
        B, ch, _ = params.shape
        assert ch in [
            1,
            3,
        ], f"Unsupported number of channels: {ch}. Expected ch=1 or ch=3."
        k = self.kernel
        mu_xy, cov_matrix, w, alpha, c, color_mean = self.extract_parameters(
            params, k, ch
        )
        # mu_xy: [B, ch, k, D=3 or D=5]
        # cov_matrix: [B, ch, k, D, D]
        # w: [B, ch, k]
        # alpha: [B, ch, k] or None
        # c: [B, ch, k] or None

        d = cov_matrix.shape[-1]  # D=3 or D=5
        eye_d = torch.eye(d, device=cov_matrix.device).view(
            1, 1, 1, d, d
        )  # [1,1,1,D,D]
        cov_matrix_reg = cov_matrix + (self.min_diag + 1e-3) * eye_d  # [B, ch, k, D, D]
        L = torch.linalg.cholesky(cov_matrix_reg)  # [B, ch, k, D, D]
        Sigma_inv = torch.cholesky_inverse(L)  # [B, ch, k, D, D]
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.transpose(-2, -1))  # [B, ch, k, D, D]

        g = self.grid(height, width, params.device)  # [W, H, 2]

        g_expanded = g.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,1,W,H,2]
        g_expanded = g_expanded.repeat(B, ch, k, 1, 1, 1)  # [B, ch, k, W, H, 2]

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_full = mu_xy.unsqueeze(-2).unsqueeze(-2)  # [B, ch, k, 1, 1, D]

            color_mean_expanded = (
                color_mean.unsqueeze(-2).unsqueeze(-2).repeat(1, 1, 1, height, width, 1)
            )
            combined_features = torch.cat(
                [g_expanded, color_mean_expanded], dim=-1
            )  # [B, ch, k, W, H, 5]

            ker = self.gaussian_cauchy_kernel(
                combined_features, mu_full, Sigma_inv, alpha, c
            )  # [B, ch, k, W, H]
        else:
            mu_spatial = mu_xy[..., :2].reshape(
                B, ch, k, 1, 1, 2
            )  # [B, ch, k, 1, 1, 2]
            Sigma_inv_spatial = Sigma_inv[..., :2, :2]  # [B, ch, k, 2, 2]
            ker = self.gaussian_kernel(
                g_expanded, mu_spatial, Sigma_inv_spatial
            )  # [B, ch, k, W, H]

        detJ = torch.det(Sigma_inv).sqrt()  # [B, ch, k]
        ker = ker * detJ.unsqueeze(-1).unsqueeze(-1)  # [B, ch, k, W, H]

        # Spatially Adaptive Weighting
        mu_spatial = mu_xy[..., :2]  # [B, ch, k, 2]
        mu_spatial_expanded = mu_spatial.unsqueeze(-2).unsqueeze(
            -2
        )  # [B, ch, k, 1, 1, 2]
        distance_sq = torch.sum(
            (g_expanded - mu_spatial_expanded) ** 2, dim=-1
        )  # [B, ch, k, W, H]

        sigma_w = 0.1  # Hyperparameter
        w_spatial = torch.exp(-distance_sq / (2 * sigma_w**2))  # [B, ch, k, W, H]
        w_spatial = F.softmax(w_spatial, dim=2)  # [B, ch, k, W, H]

        ker = ker * w_spatial  # [B, ch, k, W, H]
        ker_sum = ker.sum(dim=2, keepdim=True)  # [B, ch, 1, W, H]
        ker = ker / (ker_sum + 1e-8)  # [B, ch, k, W, H]
        out = ker.sum(dim=2)  # [B, ch, W, H]
        return torch.clamp(out, min=0.0, max=1.0)  # [B, ch, W, H]

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)


class MoE_0(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type

        self.min_diag_param = nn.Parameter(torch.log(torch.exp(torch.tensor(1e-6)) - 1))
        # self.max_inv_param = nn.Parameter(torch.log(torch.exp(torch.tensor(10.0)) - 1))
        self.min_denominator_param = nn.Parameter(
            torch.log(torch.exp(torch.tensor(1e-8)) - 1)
        )

        self.register_buffer(
            "tril_indices_2", torch.tril_indices(row=2, col=2, offset=0)
        )
        self.register_buffer(
            "tril_indices_3", torch.tril_indices(row=3, col=3, offset=0)
        )

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        return torch.stack((gx, gy), dim=-1).float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def construct_lower_triangular(
        self, params: torch.Tensor, size: int
    ) -> torch.Tensor:
        B, ch, k, n = params.shape
        indices = self.tril_indices_3 if size == 3 else self.tril_indices_2
        L = torch.zeros(B, ch, k, size, size, device=params.device, dtype=params.dtype)

        L[..., indices[0], indices[1]] = params

        diag_mask = indices[0] == indices[1]
        L[..., indices[0][diag_mask], indices[1][diag_mask]] = (
            F.softplus(L[..., indices[0][diag_mask], indices[1][diag_mask]]) + 1e-2
        )

        L[..., indices[0][diag_mask], indices[1][diag_mask]] = torch.clamp(
            L[..., indices[0][diag_mask], indices[1][diag_mask]], min=1e-3
        )

        return L

    def cov_mat(
        self,
        L_spatial: torch.Tensor,
        theta_xy: torch.Tensor,
        L_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        C_xy = R @ L_spatial @ L_spatial.transpose(-2, -1) @ R.transpose(-2, -1)
        C_color = L_color @ L_color.transpose(-2, -1)
        B, ch, k, _, _ = C_xy.shape
        C_full = torch.zeros(B, ch, k, 5, 5, device=C_xy.device, dtype=C_xy.dtype)
        C_full[..., :2, :2] = C_xy
        C_full[..., 2:, 2:] = C_color
        return C_full

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        B, _, _ = p.shape
        p = p.view(B, ch, k, -1)

        # p = F.layer_norm(p, [p.size(-1)])

        # log_parameter_stats(p, 0, 0)

        # p = torch.sigmoid(p)

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial = self.construct_lower_triangular(L_spatial_params, size=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + F.softplus(
                self.min_diag_param
            )
            L_color_params = p[..., 9:15].reshape(B, ch, k, 6)
            L_color = self.construct_lower_triangular(L_color_params, size=3)
            color_mean = p[..., 15:18].reshape(B, ch, k, 3)
            color_mean = color_mean - color_mean.mean(dim=(2, 3), keepdim=True)
            mu_xy = torch.cat([mu_x, mu_y], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            cov_matrix = cov_matrix * self.sharpening_factor
            return mu_xy, cov_matrix, w, alpha, c, color_mean, L_color
        else:
            # Similar structure for GAUSSIAN kernel_type
            mu_x = p[..., 0].reshape(B, ch, k, 1)
            mu_y = p[..., 1].reshape(B, ch, k, 1)
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)
            L_spatial = self.construct_lower_triangular(L_spatial_params, size=2)
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)
            alpha, c = None, None
            L_color_params = p[..., 7:13].reshape(B, ch, k, 6)
            L_color = self.construct_lower_triangular(L_color_params, size=3)
            color_mean = p[..., 13:16].reshape(B, ch, k, 3)
            color_mean = color_mean - color_mean.mean(dim=(2, 3), keepdim=True)
            mu_xy = torch.cat([mu_x, mu_y], dim=-1)
            cov_matrix = self.cov_mat(L_spatial, theta_xy, L_color, ch)
            cov_matrix = cov_matrix * self.sharpening_factor
            return mu_xy, cov_matrix, w, alpha, c, color_mean

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape
        k = self.cfg.kernel

        mu_xy, cov_matrix, w, alpha, c, color_mean = self.extract_parameters(
            params, k, ch
        )

        # cov_matrix = 0.5 * (cov_matrix + cov_matrix.transpose(-2, -1))

        eps = F.softplus(self.min_diag_param)
        d = cov_matrix.shape[-1]
        eye_d = torch.eye(d, device=cov_matrix.device).view(1, 1, 1, d, d)
        # cov_matrix_reg = cov_matrix + eps * eye_d

        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        min_eig = eigenvalues.min(dim=-1).values[..., None, None]
        eps_adaptive = F.softplus(-min_eig) + 1e-3
        cov_matrix_reg = cov_matrix + (eps + eps_adaptive) * eye_d

        # eigenvalues = torch.linalg.eigvalsh(cov_matrix_reg)
        # min_eig = eigenvalues.min(dim=-1, keepdim=True).values
        # singular_elements = (min_eig < 1e-3).any(dim=(-1, -2, -3))  # Batch-wise check

        # if singular_elements.any():
        #     indices = torch.where(singular_elements)
        #     print(f"Singular covariance matrices found at batch indices: {indices}")
        #     # Apply additional regularization
        #     additional_eps = 1e-2
        #     cov_matrix_reg = cov_matrix_reg + additional_eps * eye_d

        # Sigma_inv = torch.linalg.solve(cov_matrix_reg, eye_d)

        # try:
        #     Sigma_inv = torch.linalg.solve(cov_matrix_reg, eye_d)
        # except RuntimeError as e:
        #     print(
        #         f"torch.linalg.solve failed: {e}. Using pseudo-inverse as a fallback."
        #     )
        #     Sigma_inv = torch.linalg.pinv(cov_matrix_reg)

        # if torch.isnan(Sigma_inv).any() or torch.isinf(Sigma_inv).any():
        #     print("Sigma_inv contains NaNs or Infs after inversion. Clamping values.")
        #     Sigma_inv = torch.clamp(Sigma_inv, min=1e-3, max=1e3)

        L = torch.linalg.cholesky(cov_matrix_reg)
        Sigma_inv = torch.cholesky_inverse(L)
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.mT)

        # Sigma_inv = torch.clamp(
        #     Sigma_inv,
        #     min=F.softplus(self.min_diag_param),
        #     max=F.softplus(self.max_inv_param),  # Use learnable max_inv_param
        # )

        # try:
        #     L = torch.linalg.cholesky(cov_matrix)
        #     Sigma_inv = torch.cholesky_inverse(L)
        # except RuntimeError:
        #     eye = (
        #         torch.eye(cov_matrix.size(-1), device=cov_matrix.device)
        #         .unsqueeze(0)
        #         .unsqueeze(0)
        #         .unsqueeze(0)
        #     )
        #     cov_matrix += F.softplus(self.min_diag_param) * eye
        #     L = torch.linalg.cholesky(cov_matrix)
        #     Sigma_inv = torch.cholesky_inverse(L)
        # Sigma_inv = torch.clamp(Sigma_inv, min=-10.0, max=10.0)

        g = self.grid(height, width, params.device)
        g_color = torch.zeros(height, width, ch, device=params.device)
        x = torch.cat([g, g_color], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mu_full = torch.cat([mu_xy, color_mean], dim=-1).unsqueeze(3).unsqueeze(4)
        S = Sigma_inv.unsqueeze(3).unsqueeze(4)

        if self.cfg.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(x, mu_full, S, alpha, c)
        else:
            ker = self.gaussian_kernel(x, mu_full, S)

        detJ = torch.det(Sigma_inv[..., :2, :2])  # [B, ch, k]
        detJ = detJ.clamp(min=1e-3)  # Prevent division by zero or negative determinants

        # Reshape detJ for broadcasting
        detJ = detJ.unsqueeze(-1).unsqueeze(-1)  # [B, ch, k, 1, 1]

        # Apply Jacobian correction
        ker = ker * detJ

        ker = ker * w.view(B, ch, k, 1, 1)
        ker_sum = ker.sum(dim=2, keepdim=True)
        ker = ker / (ker_sum + 1e-8)
        out = ker.sum(dim=2)
        return torch.clamp(out, min=0.0, max=1.0)

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = Sigma_inv @ d.unsqueeze(-1)
        e = -0.5 * (x1 @ x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)

        norm_x = torch.linalg.norm(d[..., :2], dim=-1)
        c_e = c.unsqueeze(-1).unsqueeze(-1).expand_as(norm_x)
        diag_val = Sigma_inv[..., 0, 0].clamp(min=F.softplus(self.min_diag_param))
        denominator = c_e * diag_val
        denominator = torch.clamp(
            denominator, min=F.softplus(self.min_denominator_param)
        )
        C_csigma = 1.0 / (1.0 + norm_x**2 / denominator)
        alpha_e = alpha.unsqueeze(-1).unsqueeze(-1).expand_as(C_csigma)
        combined = alpha_e * G_sigma + (1 - alpha_e) * C_csigma
        return combined

    def gaussian_kernel(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, H, W, 5]
        d_spatial = d[..., :2]  # [B, ch, k, H, W, 2]
        x1 = d_spatial.unsqueeze(-2)  # [B, ch, k, H, W, 1, 2]
        x2 = Sigma_inv[..., :2, :2] @ d_spatial.unsqueeze(-1)  # [B, ch, k, H, W, 2, 1]
        e = -0.5 * (x1 @ x2).squeeze(-1).squeeze(-1)  # [B, ch, k, H, W]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, H, W]
        e = e - mx  # For numerical stability
        G_sigma = torch.exp(e)  # [B, ch, k, H, W]
        return G_sigma

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)


class MoE_2(Backbone[MoEConfig]):
    def __init__(self, cfg: MoEConfig):
        super().__init__(cfg)
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)
        yy = torch.linspace(0.0, 1.0, height, device=device)
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")
        return torch.stack((gx, gy), dim=-1).float()

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        R = torch.cat([ct, -st, st, ct], dim=-1)
        return R.view(*theta.shape, 2, 2)

    def cov_mat(
        self,
        scale: torch.Tensor,
        theta_xy: torch.Tensor,
        scale_color: torch.Tensor,
        rho_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)
        S = torch.diag_embed(scale)
        C_xy = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        if ch == 1:
            C_color = scale_color.squeeze(-1).squeeze(-1)
            C_full = torch.zeros(*C_xy.shape[:-2], 3, 3, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2, 2] = C_color
        elif ch == 3:
            rho = rho_color.unsqueeze(-1)
            C_rgb = torch.diag_embed(scale_color) + rho @ rho.transpose(-2, -1)
            C_full = torch.zeros(*C_xy.shape[:-2], 5, 5, device=scale.device)
            C_full[..., :2, :2] = C_xy
            C_full[..., 2:, 2:] = C_rgb
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        return C_full

    def extract_parameters(
        self, p: torch.Tensor, k: int, ch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, _ = p.shape
        # p = F.layer_norm(p, [p.size(-1)])

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2))
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            alpha = torch.sigmoid(p[:, :, 6 * k : 7 * k].reshape(B, ch, k))
            c = F.softplus(p[:, :, 7 * k : 8 * k].reshape(B, ch, k))
            if ch == 1:
                scale_color = F.softplus(p[:, :, 8 * k : 9 * k].reshape(B, ch, k, 1))
            elif ch == 3:
                scale_color = F.softplus(p[:, :, 8 * k : 11 * k].reshape(B, ch, k, 3))
            rho_color = torch.tanh(p[:, :, 11 * k : 12 * k].reshape(B, ch, k, 1))
        else:  # KernelType.GAUSSIAN
            mu_x = p[:, :, 0:k].reshape(B, ch, k, 1)
            mu_y = p[:, :, k : 2 * k].reshape(B, ch, k, 1)
            scale_xy = F.softplus(p[:, :, 2 * k : 4 * k].reshape(B, ch, k, 2))
            theta_xy = (p[:, :, 4 * k : 5 * k].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi
            w = F.softmax(p[:, :, 5 * k : 6 * k].reshape(B, ch, k), dim=-1)
            if ch == 1:
                scale_color = F.softplus(p[:, :, 6 * k : 7 * k].reshape(B, ch, k, 1))
            elif ch == 3:
                scale_color = F.softplus(p[:, :, 6 * k : 9 * k].reshape(B, ch, k, 3))
            rho_color = torch.tanh(p[:, :, 9 * k : 10 * k].reshape(B, ch, k, 1))
            alpha = None
            c = None

        mu = torch.cat([mu_x, mu_y], dim=-1)
        cov_matrix = (
            self.cov_mat(scale_xy, theta_xy, scale_color, rho_color, ch)
            * self.sharpening_factor
        )
        return mu, cov_matrix, w, alpha, c

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d = x - mu
        x1 = d.unsqueeze(-2)
        x2 = Sigma_inv @ d.unsqueeze(-1)
        e = -0.5 * (x1 @ x2).squeeze(-1).squeeze(-1)
        mx = e.max(dim=2, keepdim=True).values
        e = e - mx
        G_sigma = torch.exp(e)

        norm_x = torch.linalg.norm(d[..., :2], dim=-1)
        c_e = c.unsqueeze(-1).unsqueeze(-1).expand_as(norm_x)
        diag_val = Sigma_inv[..., 0, 0].clamp(min=1e-8)
        denominator = c_e * diag_val
        denominator = torch.clamp(denominator, min=1e-8)

        C_csigma = 1.0 / (1.0 + norm_x**2 / denominator)
        alpha_e = alpha.unsqueeze(-1).unsqueeze(-1).expand_as(C_csigma)
        combined = alpha_e * G_sigma + (1 - alpha_e) * C_csigma
        return combined

    def gaussian_kernel(
        self, x: torch.Tensor, mu: torch.Tensor, Sigma_inv: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, H, W, 5]
        d_spatial = d[..., :2]  # [B, ch, k, H, W, 2]
        x1 = d_spatial.unsqueeze(-2)  # [B, ch, k, H, W, 1, 2]
        x2 = Sigma_inv[..., :2, :2] @ d_spatial.unsqueeze(-1)  # [B, ch, k, H, W, 2, 1]
        e = -0.5 * (x1 @ x2).squeeze(-1).squeeze(-1)  # [B, ch, k, H, W]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, H, W]
        e = e - mx  # For numerical stability
        G_sigma = torch.exp(e)  # [B, ch, k, H, W]
        return G_sigma

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape
        k = self.kernel

        mu, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)
        eps = 1e-6
        d = cov_matrix.shape[-1]
        eye_d = torch.eye(d, device=cov_matrix.device).view(1, 1, 1, d, d)

        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        min_eig = eigenvalues.min(dim=-1).values[..., None, None]
        eps_adaptive = F.softplus(-min_eig) + 1e-3
        cov_matrix_reg = cov_matrix + (eps + eps_adaptive) * eye_d

        L = torch.linalg.cholesky(cov_matrix_reg)
        Sigma_inv = torch.cholesky_inverse(L)
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.mT)

        device = params.device

        g = self.grid(height, width, device)  # (height, width, 2)
        g_color = torch.zeros(
            height, width, ch, device=device
        )  # shape => (height, width, 2 + ch)
        g_full = (
            torch.cat([g, g_color], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # => (B, ch, k, height, width, 2+ch) after broadcast

        mu_color = torch.zeros(B, ch, k, ch, device=device)
        mu_full = (
            torch.cat([mu, mu_color], dim=-1).unsqueeze(3).unsqueeze(4)
        )  # => (B, ch, k, 1, 1, 2+ch)
        S = Sigma_inv.unsqueeze(3).unsqueeze(4)  # => (B, ch, k, 1, 1, d, d)

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(g_full, mu_full, S, alpha, c)
        else:
            ker = self.gaussian_kernel(g_full, mu_full, S)

        detJ = torch.det(Sigma_inv[..., :2, :2])  # [B, ch, k]
        detJ = detJ.clamp(min=1e-3)  # Prevent division by zero or negative determinants

        detJ = detJ.unsqueeze(-1).unsqueeze(-1)  # [B, ch, k, 1, 1]
        ker = ker * detJ

        ker = ker * w.view(B, ch, k, 1, 1)
        ker_sum = ker.sum(dim=2, keepdim=True)
        ker = ker / (ker_sum + 1e-8)
        out = ker.sum(dim=2)
        return torch.clamp(out, min=0.0, max=1.0)


class MoE_3(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.kernel = cfg.kernel
        self.sharpening_factor = cfg.sharpening_factor
        self.kernel_type = cfg.kernel_type
        self.min_diag = 1e-6
        self.min_denominator = 1e-8

    def grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        xx = torch.linspace(0.0, 1.0, width, device=device)  # [W]
        yy = torch.linspace(0.0, 1.0, height, device=device)  # [H]
        gx, gy = torch.meshgrid(xx, yy, indexing="ij")  # Each [W, H]
        grid = torch.stack((gx, gy), dim=-1)  # [W, H, 2]
        return grid.float()  # [W, H, 2]

    def ang_to_rot_mat(self, theta: torch.Tensor) -> torch.Tensor:
        ct = torch.cos(theta).unsqueeze(-1)  # [B, ch, k, 1]
        st = torch.sin(theta).unsqueeze(-1)  # [B, ch, k, 1]
        R = torch.cat([ct, -st, st, ct], dim=-1)  # [B, ch, k, 4]
        return R.view(*theta.shape, 2, 2)  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size1(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 1, 1, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 1, 1]
        L[..., 0, 0] = F.softplus(params[..., 0]) + 1e-2  # [B, ch, k]
        return L  # [B, ch, k, 1, 1]

    def construct_lower_triangular_size2(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 3]
        l11, l21, l22 = torch.split(params, 1, dim=-1)  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 2, 2, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 2, 2]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 2, 2]

    def construct_lower_triangular_size3(self, params: torch.Tensor) -> torch.Tensor:
        B, ch, k, _ = params.shape  # [B, ch, k, 6]
        l11, l21, l22, l31, l32, l33 = torch.split(
            params, 1, dim=-1
        )  # Each [B, ch, k, 1]
        l11 = F.softplus(l11) + 1e-2  # [B, ch, k, 1]
        l22 = F.softplus(l22) + 1e-2  # [B, ch, k, 1]
        l33 = F.softplus(l33) + 1e-2  # [B, ch, k, 1]
        L = torch.zeros(
            B, ch, k, 3, 3, device=params.device, dtype=params.dtype
        )  # [B, ch, k, 3, 3]
        L[..., 0, 0] = l11.squeeze(-1)  # [B, ch, k]
        L[..., 1, 0] = l21.squeeze(-1)  # [B, ch, k]
        L[..., 1, 1] = l22.squeeze(-1)  # [B, ch, k]
        L[..., 2, 0] = l31.squeeze(-1)  # [B, ch, k]
        L[..., 2, 1] = l32.squeeze(-1)  # [B, ch, k]
        L[..., 2, 2] = l33.squeeze(-1)  # [B, ch, k]
        return L  # [B, ch, k, 3, 3]

    def construct_lower_triangular(
        self, params: torch.Tensor, size: int
    ) -> torch.Tensor:
        if size == 1:
            return self.construct_lower_triangular_size1(params)  # [B, ch, k, 1, 1]
        elif size == 2:
            return self.construct_lower_triangular_size2(params)  # [B, ch, k, 2, 2]
        elif size == 3:
            return self.construct_lower_triangular_size3(params)  # [B, ch, k, 3, 3]
        else:
            raise ValueError(
                f"Unsupported size: {size}. Only size=1, 2, and 3 are supported."
            )

    def cov_mat(
        self,
        L_spatial: torch.Tensor,
        theta_xy: torch.Tensor,
        L_color: torch.Tensor,
        ch: int,
    ) -> torch.Tensor:
        R = self.ang_to_rot_mat(theta_xy)  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(
            R, torch.matmul(L_spatial, L_spatial.transpose(-2, -1))
        )  # [B, ch, k, 2, 2]
        C_xy = torch.matmul(C_xy, R.transpose(-2, -1))  # [B, ch, k, 2, 2]
        if ch == 1:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,1,k,1,1]
            B_, ch_, k_, _, _ = C_xy.shape
            C_full = torch.zeros(
                B_, ch_, k_, 3, 3, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,1,k,3,3]
            C_full[..., :2, :2] = C_xy  # [B,1,k,2,2]
            C_full[..., 2, 2] = C_color.squeeze(-1)  # [B,1,k]
        elif ch == 3:
            C_color = torch.matmul(L_color, L_color.transpose(-2, -1))  # [B,3,k,3,3]
            B_, ch_, k_, _, _ = C_xy.shape
            C_full = torch.zeros(
                B_, ch_, k_, 5, 5, device=C_xy.device, dtype=C_xy.dtype
            )  # [B,3,k,5,5]
            C_full[..., :2, :2] = C_xy  # [B,3,k,2,2]
            C_full[..., 2:, 2:] = C_color  # [B,3,k,3,3]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")
        return C_full * self.sharpening_factor  # [B, ch, k, D, D]

    def extract_parameters(self, p: torch.Tensor, k: int, ch: int) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        B, _, _ = p.shape  # [B, ch, k * param_per_kernel]
        p = p.view(B, ch, k, -1)  # [B, ch, k, param_per_kernel]
        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, size=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            alpha = torch.sigmoid(p[..., 7].reshape(B, ch, k))  # [B, ch, k]
            c = F.softplus(p[..., 8].reshape(B, ch, k)) + self.min_diag  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 9:10].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 9:15].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 15:18].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, size=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, alpha, c
        elif self.kernel_type == KernelType.GAUSSIAN:
            mu_x = p[..., 0].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            mu_y = p[..., 1].reshape(B, ch, k, 1)  # [B, ch, k, 1]
            L_spatial_params = p[..., 2:5].reshape(B, ch, k, 3)  # [B, ch, k, 3]
            L_spatial = self.construct_lower_triangular(
                L_spatial_params, size=2
            )  # [B, ch, k, 2, 2]
            theta_xy = (p[..., 5].reshape(B, ch, k) + torch.pi) % (
                2 * torch.pi
            ) - torch.pi  # [B, ch, k]
            w = F.softmax(p[..., 6].reshape(B, ch, k), dim=-1)  # [B, ch, k]
            if ch == 1:
                L_color_params = p[..., 7:8].reshape(B, ch, k, 1)  # [B,1,k,1]
                color_mean = torch.zeros_like(mu_x)  # [B,1,k,1]
            elif ch == 3:
                L_color_params = p[..., 7:13].reshape(B, ch, k, 6)  # [B,3,k,6]
                color_mean = p[..., 13:16].reshape(B, ch, k, 3)  # [B,3,k,3]
            else:
                raise ValueError(f"Unsupported number of channels: {ch}")
            color_cov_size = 1 if ch == 1 else 3
            L_color = self.construct_lower_triangular(
                L_color_params, size=color_cov_size
            )  # [B, ch, k, size, size]
            mu_xy = torch.cat([mu_x, mu_y, color_mean], dim=-1)  # [B, ch, k, D]
            cov_matrix = self.cov_mat(
                L_spatial, theta_xy, L_color, ch
            )  # [B, ch, k, D, D]
            return mu_xy, cov_matrix, w, None, None
        else:
            raise NotImplementedError(
                f"Kernel type {self.kernel_type} not implemented."
            )

    def gaussian_cauchy_kernel(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        Sigma_inv: torch.Tensor,
        alpha: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> torch.Tensor:
        d = x - mu  # [B, ch, k, W, H, D]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        norm_x = torch.linalg.norm(d[..., :2], dim=-1)  # [B, ch, k, W, H]
        denominator = c.unsqueeze(-1).unsqueeze(-1) * Sigma_inv[..., 0, 0].clamp(
            min=self.min_diag
        )  # [B, ch, k, 1, 1]
        denominator = denominator.clamp(min=self.min_denominator)  # [B, ch, k, 1, 1]
        C_csigma = 1.0 / (1.0 + norm_x**2 / denominator)  # [B, ch, k, W, H]
        combined = (
            alpha.unsqueeze(-1).unsqueeze(-1) * G_sigma
            + (1 - alpha.unsqueeze(-1).unsqueeze(-1)) * C_csigma
        )  # [B, ch, k, W, H]
        return combined  # [B, ch, k, W, H]

    def gaussian_kernel(
        self, x: torch.Tensor, mu_spatial: torch.Tensor, Sigma_inv_spatial: torch.Tensor
    ) -> torch.Tensor:
        d = x - mu_spatial  # [B, ch, k, W, H, 2]
        e = -0.5 * torch.einsum(
            "bckwhd,bckde,bckwhe->bckwh", d, Sigma_inv_spatial, d
        )  # [B, ch, k, W, H]
        mx = e.max(dim=2, keepdim=True).values  # [B, ch, 1, W, H]
        e = e - mx  # [B, ch, k, W, H]
        G_sigma = torch.exp(e)  # [B, ch, k, W, H]
        return G_sigma  # [B, ch, k, W, H]

    def forward_spatial(
        self, height: int, width: int, params: torch.Tensor
    ) -> torch.Tensor:
        B, ch, _ = params.shape  # [B, ch, k * param_per_kernel]
        k = self.kernel  # int
        mu_xy, cov_matrix, w, alpha, c = self.extract_parameters(params, k, ch)
        # mu_xy: [B, ch, k, D]
        # cov_matrix: [B, ch, k, D, D]
        # w: [B, ch, k]
        # alpha: [B, ch, k] or None
        # c: [B, ch, k] or None

        d = cov_matrix.shape[-1]  # D=3 or 5
        eye_d = torch.eye(d, device=cov_matrix.device).view(
            1, 1, 1, d, d
        )  # [1,1,1,D,D]

        # cov_matrix_reg = cov_matrix + (self.min_diag + 1e-3) * eye_d  # [B, ch, k, D, D]

        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        min_eig = eigenvalues.min(dim=-1).values[..., None, None]
        eps_adaptive = F.softplus(-min_eig) + 1e-8
        cov_matrix_reg = cov_matrix + (self.min_diag + eps_adaptive) * eye_d

        L = torch.linalg.cholesky(cov_matrix_reg)  # [B, ch, k, D, D]
        Sigma_inv = torch.cholesky_inverse(L)  # [B, ch, k, D, D]
        Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.transpose(-2, -1))  # [B, ch, k, D, D]

        g = self.grid(height, width, params.device)  # [W, H, 2]
        g_expanded = g.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # [1,1,1,W,H,2]
        g_expanded = g_expanded.repeat(B, ch, k, 1, 1, 1)  # [B, ch, k, W, H, 2]

        mu_full = mu_xy.unsqueeze(3).unsqueeze(4)  # [B, ch, k, 1, 1, D]

        if ch == 1:
            color_zeros = (
                torch.zeros_like(mu_xy[..., -1:]).unsqueeze(3).unsqueeze(4)
            )  # [B,1,k,1,1,1]
            color_zeros = color_zeros.expand(
                -1, -1, -1, height, width, -1
            )  # [B,1,k,H,W,1]
            x = torch.cat([g_expanded, color_zeros], dim=-1)  # [B,1,k,W,H,3]
        elif ch == 3:
            color_mean = mu_xy[..., -3:].unsqueeze(3).unsqueeze(4)  # [B,3,k,1,1,3]
            color_mean_expanded = color_mean.expand(
                -1, -1, -1, height, width, -1
            )  # [B,3,k,H,W,3]
            x = torch.cat([g_expanded, color_mean_expanded], dim=-1)  # [B,3,k,W,H,5]
        else:
            raise ValueError(f"Unsupported number of channels: {ch}")

        if self.kernel_type == KernelType.GAUSSIAN_CAUCHY:
            ker = self.gaussian_cauchy_kernel(
                x, mu_full, Sigma_inv, alpha, c
            )  # [B, ch, k, W, H]
        else:
            mu_spatial = mu_xy[..., :2].reshape(B, ch, k, 1, 1, 2)  # [B, ch, k,1,1,2]
            Sigma_inv_spatial = Sigma_inv[..., :2, :2]  # [B, ch, k,2,2]
            ker = self.gaussian_kernel(
                g_expanded, mu_spatial, Sigma_inv_spatial
            )  # [B, ch, k, W, H]

        # detJ = torch.det(Sigma_inv[..., :2, :2]).clamp(min=1e-3)  # [B, ch, k]
        # detJ = detJ.unsqueeze(-1).unsqueeze(-1)  # [B, ch, k,1,1]
        detJ = torch.det(Sigma_inv).sqrt()
        ker = ker * detJ.unsqueeze(-1).unsqueeze(-1)
        # ker = ker * detJ  # [B, ch, k, W, H]
        ker = ker * w.view(B, ch, k, 1, 1)  # [B, ch, k, W, H]
        ker_sum = ker.sum(dim=2, keepdim=True)  # [B, ch, 1, W, H]
        ker = ker / (ker_sum + 1e-8)  # [B, ch, k, W, H]
        out = ker.sum(dim=2)  # [B, ch, W, H]
        return torch.clamp(out, min=0.0, max=1.0)  # [B, ch, W, H]

    def forward(self, height: int, width: int, params: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(height, width, params)


@dataclass
class BackboneDinoCfg_v2:
    name: Literal["dino"]
    model: Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
    backbone_cfg: "BackboneResnetCfg"
    use_registers: bool = False


class BackboneDino_v2(Backbone[BackboneDinoCfg_v2]):
    def __init__(self, cfg: BackboneDinoCfg, d_in: int, d_out: int) -> None:
        super().__init__(cfg)
        self.dino = torch.hub.load("facebookresearch/dinov2", cfg.model)
        self._configure_dino_patch_embedding(d_in)
        self.resnet_backbone = BackboneResnet(cfg.backbone_cfg, d_in, d_out)
        dino_dim = self.get_dino_feature_dim()
        self.global_token_mlp = self._create_mlp(dino_dim, d_out)
        self.local_token_mlp = self._create_mlp(dino_dim, d_out)
        self.use_registers = cfg.use_registers

    def get_dino_feature_dim(self):
        feature_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }

        return feature_dims.get(self.cfg.model.replace("_lc", ""), 768)

    def _configure_dino_patch_embedding(self, d_in: int):
        old_conv = self.dino.patch_embed.proj
        params = {
            "out_channels": old_conv.out_channels,
            "kernel_size": self._ensure_tuple(old_conv.kernel_size),
            "stride": self._ensure_tuple(old_conv.stride),
            "padding": self._ensure_tuple(old_conv.padding),
            "bias": old_conv.bias is not None,
        }
        self.dino.patch_embed.proj = nn.Conv2d(d_in, **params)

    def _ensure_tuple(self, value):
        return value if isinstance(value, tuple) else tuple(value.tolist())

    def _create_mlp(self, input_dim: int, output_dim: int):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, context: "BatchedViews") -> torch.Tensor:
        resnet_features = self.resnet_backbone(context)
        b, _, h, w = context["image"].shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            new_h = (h // self.patch_size) * self.patch_size
            new_w = (w // self.patch_size) * self.patch_size
            context["image"] = context["image"][:, :, :new_h, :new_w]

        tokens = self.dino.get_intermediate_layers(context["image"], n=1)[0]
        class_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        if self.use_registers:
            register_tokens = tokens[:, -4:]
            patch_tokens = patch_tokens[:, :-4]

        global_token = self.global_token_mlp(class_token)
        local_tokens = self.local_token_mlp(patch_tokens)

        global_token = repeat(global_token, "b c -> b c h w", b=b, h=h, w=w)
        local_tokens = repeat(
            local_tokens,
            "b (h w) c -> b c (h hps) (w wps)",
            b=b,
            h=h // self.patch_size,
            hps=self.patch_size,
            w=w // self.patch_size,
            wps=self.patch_size,
        )

        if self.use_registers:
            register_tokens = self.local_token_mlp(register_tokens)
            register_tokens = repeat(
                register_tokens,
                "b r c -> b c (h hps) (w wps)",
                b=b,
                r=4,
                h=h // self.patch_size,
                hps=self.patch_size,
                w=w // self.patch_size,
                wps=self.patch_size,
            )
            return resnet_features + local_tokens + global_token + register_tokens

        return resnet_features + local_tokens + global_token

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))
