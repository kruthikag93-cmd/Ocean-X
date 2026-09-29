import torch
import torch.nn as nn


# ==========================================
# LIGHTWEIGHT DOUBLE CONVOLUTION BLOCK
# ==========================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# LIGHTWEIGHT U-NET
# ==========================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------
        # ENCODER
        # ----------------------------------

        self.enc1 = DoubleConv(1, 8)

        self.enc2 = DoubleConv(8, 16)

        # ----------------------------------
        # BOTTLENECK
        # ----------------------------------

        self.bottleneck = DoubleConv(16, 32)

        # ----------------------------------
        # POOLING
        # ----------------------------------

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ----------------------------------
        # DECODER
        # ----------------------------------

        self.up2 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(32, 16)

        self.up1 = nn.ConvTranspose2d(
            16,
            8,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(16, 8)

        # ----------------------------------
        # FINAL OUTPUT
        # ----------------------------------

        self.output = nn.Conv2d(
            8,
            1,
            kernel_size=1
        )


    # ======================================
    # FORWARD PASS
    # ======================================

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e2)
        )

        # Decoder
        d2 = self.up2(b)

        # Skip connection
        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        # Decoder
        d1 = self.up1(d2)

        # Skip connection
        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # Final prediction
        output = self.output(d1)

        return output