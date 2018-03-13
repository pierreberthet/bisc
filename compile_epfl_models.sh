#!/bin/bash
## to be run a the root folder of all the models: "/nird/home/berthetp/darpa/bisc/morphologies/hoc_combos_syn.1_0_10.allzips"

base_dir="nird/home/berthetp/darpa/bisc/morphologies/hoc_combos_syn.1_0_10.allzips/"
cd "$base_dir"
for d in L1*/; do
	cd "$base_dir""$d"
	cd mechanisms || echo "not a valid model folder"
	rm -r x86_64
	nrnivmodl >out.o
	echo "compilation done for $d"
done
echo "all completed"