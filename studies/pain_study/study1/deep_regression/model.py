"""Band-temporal regression model for Study 1."""

from __future__ import annotations

from typing import Any


def build_band_regressor(*, nn: Any, input_shape: tuple[int, int, int], config: Any) -> Any:
    n_bands, n_channels, n_times = input_shape
    temporal_filters = int(config.get("study1.deep_regression.temporal_filters", 8))
    dropout = float(config.get("study1.deep_regression.dropout", 0.25))
    kernel_size = int(config.get("study1.deep_regression.temporal_kernel_size", 15))
    kernel_size = max(3, min(kernel_size, max(3, n_times - (1 - (n_times % 2)))))
    if kernel_size % 2 == 0:
        kernel_size -= 1

    class BandTemporalRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(
                    in_channels=n_bands,
                    out_channels=temporal_filters,
                    kernel_size=(1, kernel_size),
                    padding=(0, kernel_size // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(inplace=True),
                nn.Conv2d(
                    in_channels=temporal_filters,
                    out_channels=temporal_filters,
                    kernel_size=(n_channels, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(temporal_filters),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool2d((1, 8)),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(temporal_filters * 8, temporal_filters * 2),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(temporal_filters * 2, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            features = self.features(x)
            return self.regressor(features).squeeze(-1)

    return BandTemporalRegressor()


__all__ = ["build_band_regressor"]
