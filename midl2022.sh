
################ BASELINE ################
python runner.py +task=baseline seed=0 comet_tags=[Baseline] data.label_split=0.05
#python runner.py +exp=baseline seed=0 comet_tags=[Baseline] data.label_split=0.1
#python runner.py +exp=baseline seed=0 comet_tags=[Baseline] data.label_split=0.25
#python runner.py +exp=baseline seed=0 comet_tags=[Baseline] data.label_split=0.5
#python runner.py +exp=baseline seed=0 comet_tags=[Baseline] data.label_split=1


# python runner.py +exp=acadv seed=0 comet_tags=[ADV] data.label_split=0.05
# python runner.py +exp=acadv seed=0 comet_tags=[ADV] data.label_split=0.1
# python runner.py +exp=acadv seed=0 comet_tags=[ADV] data.label_split=0.25
# python runner.py +exp=acadv seed=0 comet_tags=[ADV] data.label_split=0.5
# python runner.py +exp=acadv seed=0 comet_tags=[ADV] data.label_split=1


python runner.py +task=acssl seed=0 comet_tags=[SEMI] data.label_split=0.05
python runner.py +exp=acssl seed=0 comet_tags=[SEMI] data.label_split=0.1
python runner.py +exp=acssl seed=0 comet_tags=[SEMI] data.label_split=0.25
python runner.py +exp=acssl seed=0 comet_tags=[SEMI] data.label_split=0.5

