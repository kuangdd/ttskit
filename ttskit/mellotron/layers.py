import torch
from librosa.filters import mel as librosa_mel_fn
import librosa
import numpy as np
from .audio_processing import dynamic_range_compression, dynamic_range_decompression, griffin_lim
from .stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvNorm2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2D, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=1, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None, **kwargs):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.denoiser = None
        self.denoiser_mode = ''
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    def griffinlim_mel(self, x, n_iters=10):
        return librosa.feature.inverse.mel_to_audio(
            np.exp(x), sr=self.sampling_rate,
            n_fft=self.stft_fn.filter_length, hop_length=self.stft_fn.hop_length, win_length=self.stft_fn.win_length,
            fmin=self.mel_fmin, fmax=self.mel_fmax, power=1, n_iter=n_iters, center=True)

    def create_denoiser(self, vocoder=None, mode='zeros'):
        voc = vocoder or self.griffin_lim_
        self.denoiser_mode = mode
        self.denoiser = Denoiser(vocoder=voc, stft=self.stft_fn, mode=mode)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, ref_level_db=20, magnitude_power=1.5):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def griffin_lim(self, x, n_iters=5):
        mels = x.data.cpu().numpy()
        wavs = []
        for mel in mels:
            wav = self.griffinlim_mel(mel, n_iters)
            wavs.append(wav)
        return wavs

    def griffin_lim_(self, x, n_iters=5):
        mel_decompress = self.spectral_de_normalize(x)
        mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 100
        mel_batch = []
        for mel_one in mel_decompress:
            spec_from_mel = torch.mm(mel_one, self.mel_basis)
            spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
            spec_from_mel = spec_from_mel * spec_from_mel_scaling
            mel_batch.append(spec_from_mel)
        mel_batch = torch.cat(mel_batch, dim=0)
        wav_outputs = griffin_lim(torch.autograd.Variable(mel_batch[:, :, :-1]), self.stft_fn, n_iters)
        return wav_outputs

    def griffin_lim_denoiser(self, x, n_iters=5, denoiser_mode='zeros', denoiser_strength=0):
        wav_outputs = self.griffin_lim_(x, n_iters=n_iters)
        if self.denoiser_mode != denoiser_mode or self.denoiser is None:
            voc = lambda x: self.griffin_lim_(x, n_iters=n_iters)
            self.create_denoiser(vocoder=voc, mode=denoiser_mode)
        wav_outputs = self.denoiser(wav_outputs, denoiser_strength)
        wav_outputs = wav_outputs.squeeze(1)
        return wav_outputs


class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, vocoder, stft, mode='zeros'):
        super(Denoiser, self).__init__()
        self.vocoder = vocoder
        self.stft = stft
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=torch.float,
                device=None)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=torch.float,
                device=None)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = self.vocoder(mel_input).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised
