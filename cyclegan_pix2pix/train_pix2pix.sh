set -e
trap "exit" INT
export PYTHONPATH=../../pypi:$PYTHONPATH
for mode in "$@"; do
	if [ $mode = "float" ]; then
		cmd="python3 train.py  --name facades_pix2pix --model pix2pix --direction BtoA --n_epochs 200  --n_epochs_decay 100 --train-mode $mode"
	else
		cmd="python3 train.py --name facades_pix2pix --model pix2pix --direction BtoA --n_epochs 200  --n_epochs_decay 100 --train-mode $mode"
	fi
	suffix=qsparse/pix2pix/$mode/
	if [[ -z "${COLAB_GPU}" ]]; then
		echo "train locally: ${mode}"
		echo "$cmd"
		flagfile=$HOME/Desktop/$suffix/finish.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			$cmd --checkpoints_dir $HOME/Desktop/$suffix --dataroot $HOME/data/facades
			echo "finish" >$flagfile
		fi
	else
		echo "train in colab: ${mode}"
		echo "$cmd"
		flagfile=/content/drive/MyDrive/ModelTraining/$suffix/finish.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			mkdir -p /content/drive/MyDrive/ModelTraining/qsparse/pix2pix/
			$cmd --checkpoints_dir /content/$suffix --dataroot /content/dataset/facades
			cp -r /content/$suffix /content/drive/MyDrive/ModelTraining/$suffix # copy to google drive
			echo "finish" >$flagfile
			rm -rf /content/$suffix
		fi
	fi
done
