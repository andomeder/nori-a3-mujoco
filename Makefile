# Nori A3 MuJoCo - terminal entry points.  `make help` lists them.
# Everything runs through uv (https://docs.astral.sh/uv/); see README for the pip fallback.
SHELL := /bin/bash
UV    ?= uv
RUN   := $(UV) run --no-sync
# Pick the torch flavour for policy inference: cpu (default) or xpu (Intel Arc / oneAPI).
TORCH ?= cpu
EXTRAS := --extra video --extra gym --extra build --extra $(TORCH)
DESCRIPTION ?= ../nori_description
OUT ?= videos

-include .env
export

.PHONY: help setup setup-xpu build-model test example sdk-example policy policy-demo demo view view-policy gpu-info requirements clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup: ## create .venv from uv.lock (TORCH=cpu|xpu, default cpu)
	$(UV) sync $(EXTRAS)
	@[ -f .env ] || cp .env.example .env

setup-xpu: ## same as `make setup TORCH=xpu` (Intel Arc GPU torch wheels)
	$(MAKE) setup TORCH=xpu

build-model: ## regenerate assets/a3.xml + meshes from a nori_description checkout (DESCRIPTION=path)
	$(RUN) python scripts/build_model.py --description $(DESCRIPTION) --keep-urdf

test: ## run the test suite (headless)
	$(RUN) python -m pytest -q

example: ## real pick-and-place through the SDK API (camera, IK, gripper)
	$(RUN) python examples/sim_pick_place.py

sdk-example: ## the SDK's mock_pick_place.py against the sim, only the session line swapped
	$(RUN) python examples/sdk_mock_pick_place.py

policy: ## scripted policy through run_policy (headless, prints a report)
	$(RUN) python examples/scripted_policy.py

policy-demo: ## scripted policy, recorded to $(OUT)/scripted_policy.mp4
	@mkdir -p $(OUT)
	$(RUN) python examples/scripted_policy.py --record $(OUT)/scripted_policy.mp4 --stills $(OUT)/stills_policy

demo: ## pick-and-place video + stills to $(OUT)/ (offscreen EGL)
	@mkdir -p $(OUT)
	$(RUN) python examples/sim_pick_place.py --record $(OUT)/demo.mp4 --stills $(OUT)/stills

view: ## interactive MuJoCo viewer on the scene (GLFW window, drag joints with the mouse)
	MUJOCO_GL=glfw $(RUN) python -m mujoco.viewer --mjcf=src/nori_a3_mujoco/assets/scene.xml

view-policy: ## run the scripted policy with the live viewer attached (NORI_VIEWER=1)
	NORI_VIEWER=1 MUJOCO_GL=glfw $(RUN) python examples/scripted_policy.py

gpu-info: ## show which torch device / ONNX providers the policy code will use
	$(RUN) python -m nori_a3_mujoco.devices

requirements: ## regenerate requirements/*.txt (pip fallback) from uv.lock
	./scripts/export_requirements.sh

clean: ## remove caches and rendered output
	rm -rf .pytest_cache build $(OUT)
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
