set -e
set -u

root=weights

# aliner
mkdir -p $root/aligner_checkpoints
wget https://github.com/ai-forever/ghost-2.0/releases/download/aligner/aligner_1020_gaze_final.ckpt -P $root/aligner_checkpoints

# blender
mkdir -p $root/blender_checkpoints
wget https://github.com/ai-forever/ghost-2.0/releases/download/aligner/blender_lama.ckpt -P $root/blender_checkpoints

# mask2former
huggingface-cli download facebook/mask2former-swin-tiny-coco-instance --local-dir facebook/mask2former-swin-tiny-coco-instance

# insightface
mkdir -p $root/insightface/models/buffalo_l
wgethttps://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -P $root/insightface
unzip $root/insightface/buffalo_l.zip -d $root/insightface/models/buffalo_l

# iresnet50
wget https://github.com/ai-forever/ghost-2.0/releases/download/aligner/backbone50_1.pth -P $root

# segformer
wget https://github.com/ai-forever/ghost-2.0/releases/download/aligner/segformer_B5_ce.onnx -P $root

# stylematte
wget https://github.com/chroneus/stylematte/releases/download/weights/stylematte_synth.pth -P $root

