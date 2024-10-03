echo "SSLAlignment (Senocak)"
echo "  VGGSoundSources - Model trained on VGGSound"
python3 eval.py --testset vggss --pth_name checkpoints/sslalign.pth.tar --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/sslalign.pth.tar --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3
echo ""

echo "SLAVC (Morgado)"
echo "  VGGSoundSources - Model trained on VGGSound"
echo ""
python3 eval.py --testset vggss --pth_name checkpoints/slavc.pth --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/slavc.pth --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3
echo ""

echo "EZVSL (Morgado)"
echo "  VGGSoundSources - Model trained on VGGSound"
echo ""
python3 eval.py --testset vggss --pth_name checkpoints/ezvsl.pth --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/ezvsl.pth --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3
echo ""

echo "FNAC (-)"
echo "  VGGSoundSources - Model trained on VGGSound"
echo ""
python3 eval.py --testset vggss --pth_name checkpoints/fnac.pth --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/fnac.pth --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3
echo ""

echo "LVS (-)"
echo "  VGGSoundSources - Model trained on VGGSound"
echo ""
python3 eval.py --testset vggss --pth_name checkpoints/lvs.pth.tar --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/lvs.pth.tar --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3
echo ""

echo "SSL-TIE (-)"
echo "  VGGSoundSources - Model trained on VGGSound"
echo ""
python3 eval.py --testset vggss --pth_name checkpoints/ssltie.pth.tar --test_gt_path metadata/vggss_annotations.json --threshold 0.5 --data_dir /path/to/VGGSS
echo "  IS3 - Model trained on VGGSound"
python3 eval.py --testset is3 --pth_name checkpoints/ssltie.pth.tar --test_gt_path metadata/IS3_annotations.json --threshold 0.5 --data_dir /path/to/IS3