import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def read_npy(filename):
    """
    Reads a .npy file.
    """
    return np.load(filename,'r+')

# --- Helper: 3x3 conv with reflect padding ---
def conv3x3_r(in_c, out_c, stride=1, bias=False):
    """
    Helper function for a 3x3 convolutional layer with reflect padding.
    Commonly used in image generation to reduce border artifacts.
    """
    return nn.Conv2d(
        in_c, out_c,
        kernel_size=3, stride=stride,
        padding=1, padding_mode='reflect',
        bias=bias
    )

# --- NEW: Squeeze-and-Excitation (SE) Block ---
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    It adaptively recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Squeeze operation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # Excitation operation
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze: Global average pooling
        y = self.fc(y).view(b, c, 1, 1) # Excitation: Learn channel weights
        return x * y.expand_as(x) # Scale features with learned weights

# --- MODIFIED: Simplified Convolutional Block for Generator (internal use) ---
class GenConvBlock(nn.Module):
    """
    A simplified convolutional block for the Generator, now incorporating:
    1. A residual connection for improved gradient flow.
    2. An optional Squeeze-and-Excitation (SE) block for channel attention.
    Consists of: Conv (3x3 reflect padded) -> Instance Normalization -> ELU activation.
    """
    def __init__(self, in_c, out_c, stride=1, use_se=True):
        super().__init__()
        self.conv1 = conv3x3_r(in_c, out_c, stride=stride)
        self.norm1 = nn.InstanceNorm2d(out_c)
        self.elu = nn.ELU()
        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(out_c)

        # Optional: Add a 1x1 convolution for residual connection if in_c != out_c or stride > 1
        # This ensures dimensions match for the skip connection.
        self.residual_conv = None
        if stride != 1 or in_c != out_c:
            self.residual_conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        identity = x # Store input for residual connection

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.elu(out)

        if self.use_se:
            out = self.se_block(out)

        # Apply residual connection
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)

        # Ensure dimensions match for addition (especially after SE block or if residual_conv was needed)
        # If out and identity have different spatial sizes, interpolate identity to match out
        if out.shape[2:] != identity.shape[2:]:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)

        return out + identity # Add residual connection

# --- Simplified Decoder Block for Generator (internal use) ---
class GenDecoderBlock(nn.Module):
    """
    A simplified decoder block for the Generator's upsampling path.
    It performs upsampling using ConvTranspose2d, then concatenates with
    skip connections and applies a GenConvBlock for refinement.
    """
    def __init__(self, in_c, skip_c, out_c, use_se=True): # Added use_se parameter
        super().__init__()
        # Upsampling using ConvTranspose2d
        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        # Simplified convolutional block after concatenation
        # Input channels are 'out_c' (from upsample) + 'skip_c' (from skip connection)
        self.conv_block = GenConvBlock(out_c + skip_c, out_c, use_se=use_se) # Pass use_se to GenConvBlock

    def forward(self, x, skip_features):
        # Upsample the input features
        x = self.upsample(x)

        # Ensure spatial dimensions match before concatenation.
        # Changed mode to 'bicubic' for potentially better detail preservation.
        if x.shape[2:] != skip_features.shape[2:]:
            x = F.interpolate(x, size=skip_features.shape[2:], mode='bicubic', align_corners=False)

        # Concatenate with skip connection features
        combined_features = torch.cat([x, skip_features], dim=1)

        # Apply convolutional block for refinement
        return self.conv_block(combined_features)

# --- Initialization function (adjusted for simplified blocks) ---
def weights_init(m):
    """
    Initializes weights of convolutional and linear layers using Kaiming Normal,
    and BatchNorm/InstanceNorm layers with Normal(1.0, 0.02) for weights and 0 for biases.
    Adjusted to reflect the simplified GenConvBlock and GenDecoderBlock names.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            # Use 'relu' nonlinearity for Kaiming initialization as ELU is similar
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


# --- MODIFIED: Build ResUNet++ Model (Generator) - Simplified Internals ---
class build_resunetplusplus(nn.Module):
    """
    The ResUNet++ model architecture with simplified internal blocks for compute optimization.
    Now includes true residual connections and Squeeze-and-Excitation blocks within GenConvBlock
    for enhanced feature learning and attention.
    The 'features' parameter controls the base width of the network.
    """
    def __init__(self, in_channels=6, out_channels=1, features=32, use_se_in_gen=True):
        super().__init__()
        # Encoder Path - uses GenConvBlock for downsampling
        # Pass use_se_in_gen to GenConvBlock instances
        self.c1 = GenConvBlock(in_channels, features, stride=1, use_se=use_se_in_gen)
        self.c2 = GenConvBlock(features, features * 2, stride=2, use_se=use_se_in_gen)
        self.c3 = GenConvBlock(features * 2, features * 4, stride=2, use_se=use_se_in_gen)
        self.c4 = GenConvBlock(features * 4, features * 8, stride=2, use_se=use_se_in_gen)

        # Bottleneck Layer - a single GenConvBlock (replaces ASPP)
        self.b1 = GenConvBlock(features * 8, features * 16, stride=1, use_se=use_se_in_gen)

        # Decoder Path - uses GenDecoderBlock for upsampling and skip connections
        # Each decoder block takes (input_from_prev_decoder, skip_features, output_channels)
        # Pass use_se_in_gen to GenDecoderBlock instances
        self.d1 = GenDecoderBlock(in_c=features * 16, skip_c=features * 8, out_c=features * 8, use_se=use_se_in_gen)
        self.d2 = GenDecoderBlock(in_c=features * 8, skip_c=features * 4, out_c=features * 4, use_se=use_se_in_gen)
        self.d3 = GenDecoderBlock(in_c=features * 4, skip_c=features * 2, out_c=features * 2, use_se=use_se_in_gen)
        self.d4 = GenDecoderBlock(in_c=features * 2, skip_c=features, out_c=features, use_se=use_se_in_gen)

        # Final Output Layer - a single 1x1 convolution with Tanh activation
        self.output = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1, padding=0, bias=False),
            nn.Tanh() # Tanh activation to normalize output to -1 to +1 range
        )

        # Apply weights initialization to the entire generator model
        self.apply(weights_init)

        # Override initialization for the final convolutional layer
        if isinstance(self.output[0], nn.Conv2d):
            nn.init.xavier_uniform_(self.output[0].weight)


    def forward(self, inputs):
        # Encoder Forward Pass
        c1 = self.c1(inputs) # Encoder stage 1 output
        c2 = self.c2(c1)     # Encoder stage 2 output
        c3 = self.c3(c2)     # Encoder stage 3 output
        c4 = self.c4(c3)     # Encoder stage 4 output (deepest encoder features)

        # Bottleneck Forward Pass
        b1 = self.b1(c4)     # Bottleneck output

        # Decoder Forward Pass (upsampling and concatenating with skip connections)
        d1 = self.d1(b1, c4) # Decoder stage 1 (b1 from bottleneck, c4 from encoder)
        d2 = self.d2(d1, c3) # Decoder stage 2 (d1 from previous decoder, c3 from encoder)
        d3 = self.d3(d2, c2) # Decoder stage 3 (d2 from previous decoder, c2 from encoder)
        d4 = self.d4(d3, c1) # Decoder stage 4 (d3 from previous decoder, c1 from encoder)

        # Final Output
        return self.output(d4)

# --- blockUNet (Utility function for building blocks - for Discriminator) ---
def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    """
    Utility function to create a sequential block for U-Net architecture.
    Can be configured for convolutional (downsample) or transposed convolutional (upsample) layers.
    Modified to consistently use LeakyReLU for Discriminator stability.
    Now includes Spectral Normalization for improved GAN stability.
    """
    block = nn.Sequential()
    # Consistently use LeakyReLU for discriminator
    block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2))

    if not transposed:
        # Apply Spectral Normalization to convolutional layers in Discriminator
        block.add_module('%s_conv' % name, nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)))
    else:
        # Apply Spectral Normalization to transposed convolutional layers in Discriminator
        block.add_module('%s_tconv' % name, nn.utils.spectral_norm(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5))
    return block

# --- Discriminator ---
class Discriminator(nn.Module):
    """
    A Discriminator network (PatchGAN-like) for distinguishing real
    seismic data from generated data. Uses a series of convolutional layers
    with Spectral Normalization for improved GAN stability.
    It is designed as a Conditional Discriminator, meaning it takes both
    the input (e.g., masked seismic) and the target (real or generated seismic)
    as its input channels.
    """
    def __init__(self, nc, nf=32): # nc should be in_channels_generator + out_channels_generator
        super(Discriminator, self).__init__()

        main = nn.Sequential()

        # Initial layer: Input nc, Output nf
        # Apply Spectral Normalization
        main.add_module('layer1_conv', nn.utils.spectral_norm(nn.Conv2d(nc, nf, 4, 2, 1, bias=False)))
        main.add_module('layer1_leakyrelu', nn.LeakyReLU(0.2))
        current_out_channels = nf

        # Downsampling layers using blockUNet
        main.add_module('layer2', blockUNet(current_out_channels, current_out_channels * 2, 'layer2', transposed=False, bn=True, relu=False, dropout=False))
        current_out_channels *= 2
        main.add_module('layer3', blockUNet(current_out_channels, current_out_channels * 2, 'layer3', transposed=False, bn=True, relu=False, dropout=False))
        current_out_channels *= 2
        main.add_module('layer4', blockUNet(current_out_channels, current_out_channels * 2, 'layer4', transposed=False, bn=True, relu=False, dropout=False))
        current_out_channels *= 2
        # Optional: Add another layer for deeper discriminator
        # main.add_module('layer5', blockUNet(current_out_channels, current_out_channels * 2, 'layer5', transposed=False, bn=True, relu=False, dropout=False))
        # current_out_channels *= 2


        # Final Conv layer (no batch norm, no activation here as output is raw logit)
        main.add_module('final_conv', nn.utils.spectral_norm(nn.Conv2d(current_out_channels, 1, 4, 1, 1, bias=False)))

        self.main = main

        # Apply weights initialization to the entire discriminator model
        self.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output
