"""Training utils."""
import datetime
import os
import torch


def get_savedir(model, dataset):
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%m_%d")
    save_dir = os.path.join(
        os.environ["LOG_DIR"], date, dataset,
        model + dt.strftime('_%H_%M_%S')
    )
    os.makedirs(save_dir)
    return save_dir


def avg_both(mrs, mrrs, hits):
    """Aggregate metrics for missing lhs and rhs.

    Args:
        mrs: Dict[str, float]
        mrrs: Dict[str, float]
        hits: Dict[str, torch.FloatTensor]

    Returns:
        Dict[str, torch.FloatTensor] mapping metric name to averaged score
    """
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MR': mr, 'MRR': mrr, 'hits@[1,3,10]': h}


def format_metrics(metrics, split):
    """Format metrics for logging."""
    result = "\t {} MR: {:.2f} | ".format(split, metrics['MR'])
    result += "MRR: {:.3f} | ".format(metrics['MRR'])
    result += "H@1: {:.3f} | ".format(metrics['hits@[1,3,10]'][0])
    result += "H@3: {:.3f} | ".format(metrics['hits@[1,3,10]'][1])
    result += "H@10: {:.3f}".format(metrics['hits@[1,3,10]'][2])
    return result


def write_metrics(writer, step, metrics, split):
    """Write metrics to tensorboard logs."""
    writer.add_scalar('{}_MR'.format(split), metrics['MR'], global_step=step)
    writer.add_scalar('{}_MRR'.format(split), metrics['MRR'], global_step=step)
    writer.add_scalar('{}_H1'.format(split), metrics['hits@[1,3,10]'][0], global_step=step)
    writer.add_scalar('{}_H3'.format(split), metrics['hits@[1,3,10]'][1], global_step=step)
    writer.add_scalar('{}_H10'.format(split), metrics['hits@[1,3,10]'][2], global_step=step)


def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total

def get_khop_entities(train_examples, deleted_triples, hop):
    selected_entities = list(set(torch.cat((deleted_triples[:,0], deleted_triples[:,2])).tolist()))
    while hop > 0:
        mask  = torch.isin(train_examples[:,0], torch.Tensor(selected_entities)) | torch.isin(train_examples[:,2], torch.Tensor(selected_entities))
        hop_triples = train_examples[mask]
        add_entities = list(set(torch.cat((hop_triples[:,0], hop_triples[:,2])).tolist()))
        selected_entities += add_entities
        hop -= 1
    return selected_entities