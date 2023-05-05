# Utility function to freeze modules for the FS2 network

def freeze_postnet(net):
    net.postnet.requires_grad_(False)

def freeze_all_except_postnet(net):
    net.encoder.requires_grad_(False)
    net.duration_predictor.requires_grad_(False)
    net.pitch_predictor.requires_grad_(False)    
    net.pitch_embed.requires_grad_(False)
    net.energy_predictor.requires_grad_(False)
    net.energy_embed.requires_grad_(False)
    net.length_regulator.requires_grad_(False)
    net.decoder.requires_grad_(False)
    net.feat_out.requires_grad_(False)
    net.pitch_bottleneck.requires_grad_(False)
    net.energy_bottleneck.requires_grad_(False)
    net.duration_bottleneck.requires_grad_(False)
    net.pitch_embedding_projection.requires_grad_(False)
    net.energy_embedding_projection.requires_grad_(False)
    net.duration_embedding_projection.requires_grad_(False)
    net.decoder_in_embedding_projection.requires_grad_(False)
    net.decoder_out_embedding_projection.requires_grad_(False)