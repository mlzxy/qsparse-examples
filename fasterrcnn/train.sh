set -e
trap "exit" INT
export PYTHONPATH=../../pypi:$PYTHONPATH
for mode in "$@"; do
	if [ $mode = "float" ]; then
		cmd="python3 trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 4 --lr 1e-3 --lr_decay_step 5 --epochs 7  --train-mode $mode --r"
	else
		cmd="python3 trainval_net.py --dataset pascal_voc --net res101 --bs 1 --nw 4 --lr 1e-3 --lr_decay_step 5 --epochs 9  --train-mode $mode --r"
	fi
	suffix=qsparse/fasterrcnn/$mode/
	if [[ -z "${COLAB_GPU}" ]]; then
		echo "train locally: ${mode}"
		echo "$cmd"
		flagfile=$HOME/Desktop/$suffix/finish.txt
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			$cmd --save_dir $HOME/Desktop/$suffix --data-root $HOME/data/ --cuda
			echo "finish" >$flagfile
		fi
	else
		flagfile=/content/drive/MyDrive/ModelTraining/$suffix/finish.txt
		echo "train in colab: ${mode}"
		echo "$cmd"
		if [ -f "$flagfile" ]; then
			echo "$flagfile exists, skip."
		else
			mkdir -p /content/drive/MyDrive/ModelTraining/qsparse/fasterrcnn/
			$cmd --save_dir /content/$suffix --data-root /content/dataset/ --cuda
			cp -r /content/$suffix /content/drive/MyDrive/ModelTraining/$suffix # copy to google drive
			echo "finish" >$flagfile
			rm -rf /content/$suffix
		fi
	fi
done
