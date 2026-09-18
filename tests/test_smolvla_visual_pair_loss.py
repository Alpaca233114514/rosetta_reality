"""Tiny CPU tensors test the proposed objective, not learned policy quality."""

from dataclasses import replace

import pytest
import torch

from rosetta_reality.vla.visual_pair_loss import VisualPairContext, visual_pair_flow_loss


def _inputs():
    actions = torch.tensor([[[[-1.0, 0.0]], [[1.0, 0.0]]]])
    context = VisualPairContext(
        state=torch.full((1, 2, 3), 2.0),
        language_tokens=torch.ones(1, 2, 2, dtype=torch.int64),
        language_masks=torch.ones(1, 2, 2, dtype=torch.bool),
        noisy_actions=torch.zeros_like(actions),
        time=torch.ones(1, 2),
        episode_ids=((49, 4),),
        frame_ids=((0, 0),),
        image_sha256=(("a" * 64, "b" * 64),),
    )
    return actions, torch.ones_like(actions, dtype=torch.bool), context


def _loss(prediction, actions, valid, context, **options):
    return visual_pair_flow_loss(
        prediction,
        actions,
        valid,
        context,
        train_episode_ids=(49, 4),
        pair_weight=options.get("pair_weight", 1.0),
        minimum_action_difference=0.1,
    )


def test_state_only_solution_has_positive_contrast_and_cannot_fix_it_with_state_bias():
    actions, valid, context = _inputs()
    state_bias = torch.tensor(0.3, requires_grad=True)
    result = _loss(state_bias.expand_as(actions), actions, valid, context)
    assert result["paired_loss"].item() == pytest.approx(2.0)
    derivative = torch.autograd.grad(result["paired_loss"], state_bias)[0]
    assert derivative.item() == 0


def test_visual_gradient_points_toward_expert_action_difference():
    actions, valid, context = _inputs()
    visual_gain = torch.tensor(0.0, requires_grad=True)
    image_feature = actions.detach().clone()  # Synthetic image feature, not real pixels.
    result = _loss(-visual_gain * image_feature, actions, valid, context)
    result["loss"].backward()
    assert visual_gain.grad.item() == pytest.approx(-5.0)
    correct = _loss(-image_feature, actions, valid, context)
    reversed_direction = _loss(image_feature, actions, valid, context)
    assert correct["loss"].item() == 0
    assert reversed_direction["loss"].item() > result["loss"].item()


def test_anchor_penalizes_common_error_even_when_pair_difference_is_correct():
    actions, valid, context = _inputs()
    result = _loss(-actions + 2, actions, valid, context)
    assert result["paired_loss"].item() == 0
    assert result["anchor_loss"].item() == 4


def test_zero_weight_control_keeps_the_same_positive_target_loss():
    actions, valid, context = _inputs()
    prediction = torch.zeros_like(actions, requires_grad=True)
    result = _loss(prediction, actions, valid, context, pair_weight=0.0)
    assert torch.equal(result["loss"], result["anchor_loss"])
    assert result["paired_loss"].item() > 0


def test_action_input_shortcut_at_partial_noise_is_rejected():
    actions, valid, context = _inputs()
    # With known z=0 and t=0.5, v=-2*x_t perfectly recovers targets without images.
    partial_noise = 0.5 * actions
    shortcut = -2 * partial_noise
    assert torch.equal(shortcut, context.noisy_actions - actions)
    context = replace(context, noisy_actions=partial_noise, time=torch.full((1, 2), 0.5))
    with pytest.raises(ValueError, match="time 1"):
        _loss(shortcut, actions, valid, context)


@pytest.mark.parametrize("field", ["state", "language_tokens", "language_masks", "noisy_actions"])
def test_nonvisual_condition_differences_cannot_masquerade_as_visual_grounding(field):
    actions, valid, context = _inputs()
    tensor = getattr(context, field).clone()
    if tensor.dtype == torch.bool:
        tensor[:, 1, 0] = False
    else:
        tensor[:, 1] += 1
    with pytest.raises(ValueError, match="identical"):
        _loss(-actions, actions, valid, replace(context, **{field: tensor}))


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_ids": ((49, 31),)},
        {"episode_ids": ((49, 49),)},
        {"frame_ids": ((0, 100),)},
        {"image_sha256": (("a" * 64, "a" * 64),)},
    ],
)
def test_hidden_or_unmatched_sample_identities_are_rejected(changes):
    actions, valid, context = _inputs()
    with pytest.raises(ValueError, match="distinct train"):
        _loss(-actions, actions, valid, replace(context, **changes))


def test_padding_cannot_create_or_dominate_supervision():
    actions, valid, context = _inputs()
    valid[:, :, :, 1] = False
    actions[:, :, :, 1] = torch.tensor([[-1000.0, 1000.0]]).unsqueeze(-1)
    prediction = -actions.clone()
    prediction = prediction + torch.tensor([0.0, 9999.0])
    result = _loss(prediction, actions, valid, context)
    assert result["loss"].item() == 0
    actions[:, :, :, 0] = 0
    with pytest.raises(ValueError, match="minimum action contrast"):
        _loss(prediction, actions, valid, context)
    valid.zero_()
    with pytest.raises(ValueError, match="shared non-padding"):
        _loss(prediction, actions, valid, context)


def test_labels_and_noise_do_not_receive_gradients_and_inputs_are_unchanged():
    actions, valid, context = _inputs()
    actions.requires_grad_()
    context.noisy_actions.requires_grad_()
    prediction = torch.zeros_like(actions, requires_grad=True)
    before = actions.detach().clone()
    result = _loss(prediction, actions, valid, context)
    result["loss"].backward()
    assert prediction.grad is not None
    assert actions.grad is None and context.noisy_actions.grad is None
    assert torch.equal(actions, before)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_invalid_treatment_weight_is_rejected(weight):
    actions, valid, context = _inputs()
    with pytest.raises(ValueError, match="pair_weight"):
        _loss(-actions, actions, valid, context, pair_weight=weight)


def test_finite_but_overflowing_values_are_not_accepted_as_training_loss():
    actions, valid, context = _inputs()
    with pytest.raises(FloatingPointError, match="non-finite"):
        _loss(torch.full_like(actions, 1e30), actions, valid, context)
