set -e
trap "exit" INT
export PYTHONPATH=../../pypi:$PYTHONPATH
for mode in "$@"; do
	if [ $mode = "float" ]; then
		cmd="python3 train.py --name horse2zebra_cyclegan --model cycle_gan  --n_epochs 200 --train-mode $mode"
	else
		cmd="python3 train.py --name horse2zebra_cyclegan --model cycle_gan --n_epochs 200 --train-mode $mode"
	fi
	suffix=qsparse/cyclegan/$mode/

	if [[ -z "${COLAB_GPU}" ]]; then
		echo "train locally: ${mode}"
		echo "$cmd"
		flagfile=$HOME/Desktop/$suffix/finish.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			$cmd --checkpoints_dir $HOME/Desktop/$suffix --dataroot $HOME/data/horse2zebra
			echo "finish" >$flagfile
		fi
	else
		flagfile=/content/drive/MyDrive/ModelTraining/$suffix/finish.txt
		echo "train in colab: ${mode}"
		echo "$cmd"
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			mkdir -p /content/drive/MyDrive/ModelTraining/qsparse/cyclegan/
			$cmd --checkpoints_dir /content/$suffix --dataroot /content/dataset/horse2zebra
			cp -r /content/$suffix /content/drive/MyDrive/ModelTraining/$suffix # copy to google drive
			echo "finish" >$flagfile
			rm -rf /content/$suffix
		fi
	fi
done
