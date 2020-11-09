import os
import torch
import logging

logger = logging.getLogger(__name__)


def weighted_mse_loss(output, target, target_weights):
    # logging.info(f"target_weights.shape = {target_weights.shape}")
    # logging.info(f"output.shape = {output.shape}")
    # logging.info(f"target.shape = {target.shape}")

    return torch.sum(target_weights * (output.squeeze(2) - target) ** 2)


def write_predictions(predictions, token_ids, output_dir, num_actual_predict_examples, tokenizer):
    with open(os.path.join(output_dir, "test_results.tsv"), "w") as writer:
        num_written_lines = 0
        logging.info("***** Predict results *****")
        for (i, logits) in enumerate(predictions):
            tokens = tokenizer.convert_ids_to_tokens(token_ids[i])
            if i >= num_actual_predict_examples:
                break
            output_line = '\t'.join(['{0} {1:.5f}'.format(t, w) for (t, w) in zip(tokens, logits)
                                     if t != '[PAD]' and t != '[CLS]'])
            writer.write(output_line)
            writer.write('\n')
            num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples
