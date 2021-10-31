set -e
trap "exit" INT
export PYTHONPATH=../../pypi:$PYTHONPATH
for mode in "$@"; do
	if [ $mode = "float" ]; then
		cmd="python3 test.py  --name horse2zebra_cyclegan --model cycle_gan  --train-mode $mode"
	else
		cmd="python3 test.py --name horse2zebra_cyclegan --model cycle_gan  --train-mode $mode"
	fi
	suffix=qsparse/cyclegan/$mode/
	if [[ -z "${COLAB_GPU}" ]]; then
		echo "test locally: ${mode}"
		echo "$cmd"
		flagfile=$HOME/Desktop/$suffix/finish.test.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			$cmd --checkpoints_dir $HOME/Desktop/$suffix --results_dir $HOME/Desktop/$suffix --dataroot $HOME/data/horse2zebra
			echo "finish" >$flagfile
		fi
	else
		echo "test in colab: ${mode}"
		echo "$cmd"
		flagfile=/content/drive/MyDrive/ModelTraining/$suffix/finish.test.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			mkdir -p /content/drive/MyDrive/ModelTraining/qsparse/cyclegan/
			$cmd --checkpoints_dir /content/$suffix --results_dir /content/$suffix --dataroot /content/dataset/horse2zebra
			cp -r /content/$suffix /content/drive/MyDrive/ModelTraining/$suffix # copy to google drive
			echo "finish" >$flagfile
			rm -rf /content/$suffix
		fi
	fi
done
