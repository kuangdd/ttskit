from tensorboardX import SummaryWriter

from .plotting import plot_waveform_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = hp.audio.sampling_rate
        self.is_first = True

    def log_training(self, g_loss, d_loss, step):
        self.add_scalar('train.g_loss', g_loss, step)
        self.add_scalar('train.d_loss', d_loss, step)

    def log_validation(self, g_loss, d_loss, generator, discriminator, target, prediction, step):
        self.add_scalar('validation.g_loss', g_loss, step)
        self.add_scalar('validation.d_loss', d_loss, step)

        self.add_audio('raw_audio_predicted', prediction, step, self.sample_rate)
        self.add_image('waveform_predicted', plot_waveform_to_numpy(prediction), step)

        self.log_histogram(generator, step)
        self.log_histogram(discriminator, step)

        if self.is_first:
            self.add_audio('raw_audio_target', target, step, self.sample_rate)
            self.add_image('waveform_target', plot_waveform_to_numpy(target), step)
            self.is_first = False

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)
