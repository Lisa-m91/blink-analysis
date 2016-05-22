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
		./extract.py --expansion=2 $infiles > $outdir/$ds.pickle
	end;
end;
