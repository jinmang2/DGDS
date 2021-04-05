import os
import json
import argparse
from collections import defaultdict

from datasets import load_dataset


DOC_DOMAIN_SPLIT = "train"


def text2line(text):
    return text.replace("\n", "\t").replace("\r", "\t").strip()


def btag(tag, text):
    return "<{}>\t{}".format(tag, text2line(text))


def load_doc2dial(args):
    doc_dataset = load_dataset(
        path=args.datasets,
        name="document_domain",
        split=DOC_DOMAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    if args.split == 'test':
        with open(args.test_data) as f:
            test = json.load(f)
        dial_dataset = []
        for domain, v1 in test['dial_data'].items():
            for title, v2 in v1.items():
                for i in v2:
                    dial_dataset.append(i)
    else:
        dial_dataset = load_dataset(
            path=args.datasets,
            name="dialogue_domain",
            split=args.split,
            cache_dir=args.cache_dir,
            ignore_verifications=True,
        )
    return doc_dataset, dial_dataset


def generate_data_4_cross_encoder(args):
    """
    Cross Encoder의 input 형태로 source/target을 제작

    Source 형태:
        last_uttr + [SEP] + dial_context + [SEP] + doc_context
    Target 형태:
        next_uttr if train or valid else ""
    """
    doc_dataset, dial_dataset = load_doc2dial(args)

    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span

    source, target = [], []
    for ex in dial_dataset:
        doc_id = ex['doc_id']
        d_doc_spans = d_doc[doc_id]
        dial_context = []
        # TODO: train/valid/test 1 loop에 다 넣기
        if args.split == "test":
            last_turn = [btag("last_turn", ex["turns"][-1]["utterance"])]
            dial_context = [
                btag(turn["role"], turn["utterance"]) for turn in ex["turns"][::-1]
            ]
            if args.full_doc:
                doc_context = [
                    btag("title", doc_id),
                    btag("doc_context", d_doc[doc_id]["doc_text"])
                ]
            else:
                pass # NotImplemented
            contexts = last_turn + dial_context + doc_context
            source.append("\t".join(contexts))
            target.append("")
            continue

        for turn in ex["turns"]:
            if not turn["references"]:
                # this task only uses instances and evaluates on the grounded turns
                continue
            utterance = text2line(turn["utterance"])
            utterance_context = btag(turn["role"], utterance)
            if turn["role"] in args.role:
                # add previous utterance as tagged query context
                last_turn = [
                    btag("last_turn", dial_context[-1].split("\t", 1)[-1])]
                # add dialog history in reverse order as tagged dialogue context
                dial_context = dial_context[::-1]
                if args.full_doc:
                    # add entire document as tagged document context
                    doc_context = [
                        btag("title", ex["doc_id"]),
                        btag("doc_context", d_doc[doc_id]["doc_text"])
                    ]
                else:
                    pass # NotImplemented
                contexts = last_turn + dial_context + doc_context
                source.append("\t".join(contexts))
                target.append(utterance)
            dial_context.append(utterance_context)
    assert len(source) == len(target), \
        "Need to ensure that source and target are same sized."
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    source_fn = os.path.join(args.output_dir, "{}.source".format(args.split))
    target_fn = os.path.join(args.output_dir, "{}.target".format(args.split))
    with open(source_fn, "w", encoding="utf-8") as fp:
        fp.write("\n".join(source))
        fp.close()
    with open(target_fn, "w", encoding="utf-8") as fp:
        fp.write("\n".join(target))
        fp.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split is 'train', 'validation' or 'test'",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default='',
        help="Test Data directory if split == 'test' only",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default='doc2dial',
        help="The root folder of your `datasets` source code",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='',
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="agent",
        help="which role's utterance for generation",
    )
    parser.add_argument(
        "--full_doc",
        type=bool,
        default=True,
        help="whether use entire document",
    )
    parser.add_argument(
        "--include_da",
        type=bool,
        default=False,
        help="whether to include DA as input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="seq2seq_output",
        help="path to output the data files",
    )

    args = parser.parse_args()
    if args.split.lower() in ['validation', 'val']:
        args.split = 'valid'
    generate_data_4_cross_encoder(args)


if __name__ == '__main__':
    main()
