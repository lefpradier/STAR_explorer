c=0
for star_type in discrete continuous; do
    for max_dist in 20 10000 ; do
        c=$((c+1))
        python3 processing_matriceadj_freq.py $max_dist $star_type $max_dist $c
    done
done
