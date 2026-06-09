# Gram Critic — the whole pipeline is config-driven; this Makefile is the readable surface.
# Override any config:  make train CRITIC_CONFIG=configs/my_critic.yaml
PY ?= python

ZOO_CONFIG    ?= configs/zoo.yaml
CRITIC_CONFIG ?= configs/critic.yaml
DEPLOY_CONFIG ?= configs/deploy_c09.yaml
VIZ_CONFIG    ?= configs/viz.yaml

.PHONY: help data zoo train deploy deploy-c06 ablation viz all clean

help:
	@echo "Gram Critic pipeline:"
	@echo "  make data        download MNIST"
	@echo "  make zoo         generate the velocity-field ΔK dataset   ($(ZOO_CONFIG))"
	@echo "  make train       fit the Gram Critic                      ($(CRITIC_CONFIG))"
	@echo "  make deploy      headline log-Euclidean table at c=0.9     ($(DEPLOY_CONFIG))"
	@echo "  make deploy-c06  same at c=0.6"
	@echo "  make ablation    oracle deliveries through the Gram        (configs/ablation.yaml)"
	@echo "  make viz         render activation animations              ($(VIZ_CONFIG))"
	@echo "  make all         data -> zoo -> train -> deploy"
	@echo "  make clean       remove runs/"

data:
	$(PY) scripts/download_mnist.py

zoo:
	$(PY) -m gram_critic zoo $(ZOO_CONFIG)

train:
	$(PY) -m gram_critic train $(CRITIC_CONFIG)

deploy:
	$(PY) -m gram_critic deploy $(DEPLOY_CONFIG)

deploy-c06:
	$(PY) -m gram_critic deploy configs/deploy_c06.yaml

ablation:
	$(PY) -m gram_critic ablation configs/ablation.yaml

viz:
	$(PY) -m gram_critic viz $(VIZ_CONFIG)

all: data zoo train deploy

clean:
	rm -rf runs/
