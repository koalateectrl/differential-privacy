#!/bin/bash

declare -i num_teachers=5
for ((i=0; i<$num_teachers; i++))
do
echo $i
python train_teachers.py --nb_teachers=$num_teachers --teacher_id=$i --dataset=mnist
done

python aggregate_teachers.py --nb_teachers=$num_teachers
python train_student_PATEG.py --nb_teachers=$num_teachers --dataset=mnist --count=50
