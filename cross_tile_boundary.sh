for ((i=2; i<=10; i++))
do
    python training_procedure.py --train --gpu_id 1 --dataset_name David_MMSys_18 --model_name perceiver6 --m_window 15 --h_window 25 --init_window 30 --provided_videos --bs 128 --lr 1e-4 --tile_h $i --tile_w $((2 * i)) --epochs 100
    python training_procedure.py --evaluate --gpu_id 1 --dataset_name David_MMSys_18 --model_name perceiver6 --m_window 15 --h_window 25 --init_window 30 --provided_videos --bs 128 --lr 1e-4 --tile_h $i --tile_w $((2 * i)) --epochs 100
    python training_procedure.py --train --gpu_id 1 --dataset_name David_MMSys_18 --model_name perceiver6 --m_window 15 --h_window 25 --init_window 30 --provided_videos --bs 128 --lr 1e-4 --tile_h $i --tile_w $((2 * i)) --use_cross_tile_boundary_loss --epochs 100
    python training_procedure.py --evaluate --gpu_id 1 --dataset_name David_MMSys_18 --model_name perceiver6 --m_window 15 --h_window 25 --init_window 30 --provided_videos --bs 128 --lr 1e-4 --tile_h $i --tile_w $((2 * i)) --use_cross_tile_boundary_loss --epochs 100
done