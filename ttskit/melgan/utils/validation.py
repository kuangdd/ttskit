import tqdm
import torch


def validate(hp, args, generator, discriminator, valloader, writer, step):
    generator.eval()
    discriminator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    for mel, audio in loader:
        mel = mel.cuda()
        audio = audio.cuda()

        # generator
        fake_audio = generator(mel)
        disc_fake = discriminator(fake_audio[:, :, :audio.size(2)])
        disc_real = discriminator(audio)
        loss_g = 0.0
        loss_d = 0.0
        for (feats_fake, score_fake), (feats_real, score_real) in zip(disc_fake, disc_real):
            loss_g += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
            for feat_f, feat_r in zip(feats_fake, feats_real):
                loss_g += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))
            loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
            loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

    loss_g_avg = loss_g_sum / len(valloader.dataset)
    loss_d_avg = loss_d_sum / len(valloader.dataset)

    audio = audio[0][0].cpu().detach().numpy()
    fake_audio = fake_audio[0][0].cpu().detach().numpy()

    writer.log_validation(loss_g_avg, loss_d_avg, generator, discriminator, audio, fake_audio, step)

    torch.backends.cudnn.benchmark = True
