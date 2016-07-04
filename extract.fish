set in_prefix "/mnt/delphi-scratch/kjw53/data/Lisa-Maria/532 nm Excitation_(Gerbera)"
set out_prefix "/mnt/delphi-scratch/kjw53/data/Lisa-Maria/"
set datasets "160131_EosHaloJF646/EosHaloJF646" "160131_EosHaloJF646/EosHalo" "160507_EH(646)_532_Trolox/EH646" "160507_EH(646)_532_Trolox/EH"

for dataset in $datasets;
	set outdir $out_prefix/(echo $dataset | sed "s@/@_@g")
	mkdir -p $outdir
	for video in $in_prefix/$dataset/*;
		# Use find as sometimes nested by date
		set infiles (find $video -name "*.tif" | sort -V)
		set ds (basename $video | grep -Eo "DS[0-9]+")
		set outfile $outdir/$ds.pickle
		set -e exclude

		switch $dataset
		case "160507_EH(646)_532_Trolox/EH646"
			switch $ds
			case DS1
				set exclude 8500-9100
			case DS2
				set exclude 0-200
			end
		case "160507_EH(646)_532_Trolox/EH"
			switch $ds
			case DS4
				set exclude 0-5000 6500-8000 8100-10001
			end
		end

		./extract.py $infiles --spot-size 2 5 --threshold 0.1 --normalize --exclude $exclude > $outdir/$ds.pickle
	end;
end;

set datasets "EH(JF646)_561_2mMTrol/EHJF646_NormalLinker_WithTrolox" "EH(JF646)_561_2mMTrol/EH Normal Linker_WithTrolox"

for dataset in $datasets;
	set outdir $out_prefix/(echo $dataset | sed "s@/@_@g")
	mkdir -p $outdir
	for video in $in_prefix/$dataset/*;
		# Use find as sometimes nested by date
		set infiles (find $video -name "*.tif" | sort -V)
		set ds (basename $video | grep -Eo "DS[0-9]+")
		set outfile $outdir/$ds.pickle

    ./extract.py $infiles --spot-size 2 5 --threshold 50 > $outdir/$ds.pickle
	end;
end;
