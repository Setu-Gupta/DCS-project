for plot in ./plot[1,2,3]
do
        cd $plot;
        for decoder in ./*
        do
                cd $decoder;
                echo "Running $plot/$decoder";
                python3 csv_gen.py 17 ./out.csv;
                cd ..;
        done
        cd ..;
done
