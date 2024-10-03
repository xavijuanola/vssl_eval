echo "SSLAlignment (Senocak)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/sslalign.pth.tar --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.3175 --data_dir /path/to/IS3
echo ""

echo "SLAVC (Morgado)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/slavc.pth --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.5450 --data_dir /path/to/IS3
echo ""

echo "EZVSL (Morgado)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/ezvsl.pth --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.6382 --data_dir /path/to/IS3
echo ""

echo "FNAC (-)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/fnac.pth --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.5538 --data_dir /path/to/IS3
echo ""

echo "LVS (-)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/lvs.pth.tar --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.6077 --data_dir /path/to/IS3
echo ""

echo "SSL-TIE (-)"
echo "IS3 - Model trained on VGGSound"
python3 infer.py --testset is3 --pth_name checkpoints/ssltie.pth.tar --test_gt_path /media/v/IS3Dataset/IS3_annotations.json --threshold 0.3338 --data_dir /path/to/IS3