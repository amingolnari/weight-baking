"""Build the probe words and sentences read by the ablation suite.

    python scripts/build_data.py --wordnet --per-category 250 --wikitext

It writes four files to --out-dir:

    probe_words.txt            word<TAB>category, ~250 words per category from WordNet (the first 8 categories are the core set)
    calibration_sentences.txt  sentences from the WikiText-103 validation split, used to fit the closed-form weight edit
    heldout_sentences.txt      sentences from the WikiText-103 test split (same domain as calibration)
    new_domain_sentences.txt   template-generated code / medical / legal / chat / finance sentences (no real-data option)

Without --wordnet the short built-in placeholder word lists are used; without --wikitext the calibration and held-out sentences are
template-generated placeholders. The first run needs internet access (WordNet corpus, WikiText-103 download). Lines starting with
'#' are comments and are ignored by the loader. Requires nltk and wordfreq (--wordnet) and datasets (--wikitext).
"""

import argparse
import itertools
import os
import random
import re

# ---- word lists and WordNet roots ----
# Built-in placeholder lists (used only without --wordnet) and the WordNet synsets that define each category.
CORE = {
    "animal": "dog cat horse eagle shark lion tiger wolf rabbit whale monkey snake owl goat deer bear fox zebra giraffe elephant camel donkey sheep pig cow mouse rat squirrel beaver otter dolphin penguin parrot sparrow pigeon swan duck goose frog lizard turtle crocodile spider ant bee butterfly salmon trout octopus crab lobster kangaroo koala panda leopard cheetah hyena buffalo moose",
    "fruit": "apple banana lemon cherry grape mango peach pear plum melon apricot coconut papaya lime fig strawberry pineapple raspberry blueberry blackberry watermelon grapefruit tangerine nectarine pomegranate guava lychee persimmon cranberry kiwi avocado currant gooseberry mulberry quince tamarind passionfruit dragonfruit cantaloupe clementine mandarin kumquat plantain elderberry boysenberry jackfruit durian starfruit olive citron",
    "vehicle": "car truck bus train bicycle airplane boat tram van scooter taxi ship helicopter motorcycle subway ambulance tractor canoe ferry jeep limousine minivan sedan coupe wagon trolley locomotive yacht sailboat kayak raft submarine glider rocket spaceship hovercraft bulldozer forklift snowmobile sled carriage chariot rickshaw moped tricycle unicycle skateboard gondola barge tanker freighter trawler hatchback convertible pickup",
    "tool": "hammer wrench saw drill screwdriver chisel pliers shovel axe knife ladder scissors rake crowbar pickaxe sandpaper trowel clamp mallet spanner hacksaw hatchet sickle scythe hoe spade pitchfork tongs tweezers awl vise anvil lathe sander grinder ratchet level ruler caliper plunger funnel whisk spatula ladle grater peeler corkscrew stapler wheelbarrow bellows soldering crimper router jigsaw",
    "color": "red green yellow purple violet pink brown black white gray beige crimson turquoise maroon scarlet indigo magenta teal cyan lavender amber ivory khaki navy olive-green burgundy coral mauve ochre fuchsia chartreuse vermilion cerulean taupe sepia azure emerald-green lilac peach-pink mustard salmon-pink charcoal cream tan aquamarine periwinkle plum-purple russet saffron",
    "country": "France Japan Brazil Egypt Canada India China Chile Kenya Norway Spain Italy Mexico Peru Sweden Poland Vietnam Nigeria Greece Ireland Germany Portugal Belgium Austria Denmark Finland Iceland Hungary Romania Bulgaria Turkey Iran Iraq Israel Jordan Pakistan Nepal Thailand Malaysia Indonesia Philippines Australia Argentina Colombia Venezuela Ecuador Bolivia Uruguay Cuba Jamaica Morocco Tunisia Ethiopia Ghana Tanzania Uganda",
    "emotion": "joy anger fear sadness envy pride shame hope grief love happiness rage terror sorrow jealousy guilt delight anxiety despair gratitude affection contempt disgust surprise boredom loneliness nostalgia relief regret remorse resentment sympathy compassion admiration awe curiosity excitement frustration embarrassment disappointment contentment serenity melancholy dread panic euphoria bitterness tenderness longing optimism pessimism irritation amusement indignation",
    "profession": "doctor teacher lawyer engineer nurse pilot chef farmer judge baker dentist plumber carpenter journalist architect surgeon librarian electrician accountant sailor mechanic pharmacist veterinarian firefighter policeman soldier scientist professor translator photographer musician painter sculptor actor director editor programmer banker cashier waiter butcher tailor barber florist jeweler miner welder mason gardener fisherman secretary receptionist psychologist geologist astronomer",
}
EXTRA = {
    "sport": "football tennis basketball baseball cricket rugby golf hockey volleyball badminton swimming boxing wrestling fencing archery rowing sailing skiing snowboarding skating cycling gymnastics judo karate taekwondo handball squash polo lacrosse softball surfing diving marathon triathlon weightlifting curling bowling billiards darts climbing",
    "instrument": "piano violin guitar flute trumpet drum cello harp clarinet saxophone trombone oboe bassoon tuba banjo mandolin ukulele accordion harmonica xylophone tambourine organ viola harpsichord bagpipe sitar lute piccolo cymbal marimba synthesizer bugle cornet zither recorder glockenspiel bongo didgeridoo fiddle",
    "clothing": "shirt trousers jacket coat dress skirt sweater scarf glove hat sock shoe boot sandal belt tie blouse jeans shorts vest cardigan hoodie raincoat overcoat pyjamas robe apron uniform kimono sari tuxedo waistcoat stocking mitten cap bonnet shawl cloak poncho legging",
    "furniture": "chair table sofa bed desk shelf cabinet wardrobe drawer bench stool couch armchair bookcase cupboard dresser nightstand crib cot hammock ottoman futon recliner sideboard bureau chest rack stand counter pew lectern throne mattress footstool trestle settee divan cradle bunk workbench",
    "vegetable": "carrot potato onion cabbage lettuce spinach broccoli cauliflower celery cucumber tomato pepper eggplant zucchini pumpkin radish turnip beetroot parsnip leek garlic asparagus artichoke kale chard okra pea bean lentil corn squash yam shallot scallion fennel rhubarb watercress endive kohlrabi rutabaga",
    "building": "house church castle tower temple mosque palace hospital school library museum theater stadium factory warehouse barn cottage mansion skyscraper cathedral chapel monastery fortress lighthouse windmill garage hotel hostel prison courthouse bakery pharmacy greenhouse pavilion bungalow villa cabin hut shed observatory",
    "flower": "rose tulip daisy lily orchid sunflower daffodil carnation iris poppy marigold jasmine peony hyacinth magnolia camellia chrysanthemum dahlia geranium begonia petunia pansy snapdragon foxglove bluebell buttercup dandelion lotus azalea hibiscus gardenia freesia anemone zinnia primrose crocus lupin aster hydrangea wisteria",
    "metal": "iron copper gold silver tin lead zinc nickel aluminum titanium platinum mercury tungsten cobalt chromium manganese magnesium sodium potassium calcium lithium uranium plutonium bronze brass steel pewter palladium rhodium iridium osmium bismuth antimony cadmium gallium indium barium strontium beryllium vanadium",
}
WORDNET = {  # category -> WordNet synsets whose hyponym closure supplies the words
    "animal": ["animal.n.01"],
    "fruit": ["edible_fruit.n.01"],
    "vehicle": ["vehicle.n.01"],
    "tool": ["tool.n.01", "hand_tool.n.01"],
    "color": ["chromatic_color.n.01", "achromatic_color.n.01"],
    "country": ["country.n.02"],
    "emotion": ["emotion.n.01", "feeling.n.01"],
    "profession": ["professional.n.01", "skilled_worker.n.01", "craftsman.n.02"],
    "sport": ["sport.n.01"],
    "instrument": ["musical_instrument.n.01"],
    "clothing": ["garment.n.01"],
    "furniture": ["furniture.n.01"],
    "vegetable": ["vegetable.n.01"],
    "building": ["building.n.01"],
    "flower": ["flower.n.01"],
    "metal": ["metallic_element.n.01", "alloy.n.01"],
}


# ---- helper functions ----
def placeholder_words():
    """The built-in placeholder lists: category -> list of words."""
    return {c: w.split() for c, w in {**CORE, **EXTRA}.items()}


def wordnet_words(per_category):
    """Up to per_category single-word members per category from WordNet, most frequent first when wordfreq is installed."""
    import nltk

    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet")
        from nltk.corpus import wordnet as wn
    try:
        from wordfreq import zipf_frequency as zf
    except ImportError:
        zf = lambda w, lang: 0.0
    out = {}
    for cat, roots in WORDNET.items():
        seen, cand = set(), []
        for r in roots:
            s = wn.synset(r)
            nodes = list(s.closure(lambda x: x.hyponyms() + x.instance_hyponyms()))
            for node in nodes:
                for lemma in node.lemma_names():
                    if re.fullmatch(r"[A-Za-z]{3,14}", lemma) and lemma.lower() not in seen:
                        seen.add(lemma.lower())
                        cand.append(lemma if cat == "country" else lemma.lower())
        cand.sort(key=lambda w: -zf(w.lower(), "en"))
        out[cat] = cand[:per_category]
    return out


def fill(templates, slots, n, rng):
    """n distinct sentences from templates whose {slot} fields are filled from the slot lists."""
    out = set()
    for t in templates:
        names = re.findall(r"{(\w+)}", t)
        for combo in itertools.product(*[slots[k] for k in names]):
            out.add(t.format(**dict(zip(names, combo))))
    out = sorted(out)
    rng.shuffle(out)
    assert len(out) >= n, f"only {len(out)} sentences available, {n} requested"
    return out[:n]


# ---- sentence templates for the new-domain set ----
GENERAL_T = [
    "The {who} walked to the {place} {when} and found that the {thing} had already been moved.",
    "{When}, the {who} from the {place} explained why the {thing} mattered to everyone in the town.",
    "Nobody at the {place} expected the {who} to return the {thing} {when}.",
    "According to the {who}, the {thing} near the {place} was built long before the war.",
    "The {who} spent most of the afternoon at the {place}, looking for the missing {thing}.",
    "After the storm, the {who} repaired the {thing} and carried it back to the {place}.",
    "It took the {who} three weeks to finish the report about the {thing} at the {place}.",
    "The {thing} that the {who} left at the {place} was still there {when}.",
]
GENERAL_S = {
    "who": "teacher farmer mayor nurse student driver baker engineer musician reporter gardener sailor".split(),
    "place": "market station harbor library village square bridge school factory museum hospital river".split(),
    "when": [
        "yesterday morning",
        "last winter",
        "before sunrise",
        "on Friday evening",
        "two days later",
        "at the end of the month",
    ],
    "When": [
        "Yesterday morning",
        "Last winter",
        "Before sunrise",
        "On Friday evening",
        "Two days later",
        "At the end of the month",
    ],
    "thing": "bell map clock boat letter wagon lantern fence statue ledger piano window".split(),
}
DOMAINS = {
    "code": (
        [
            "The function {fn} returns {ret} when the {arg} argument is {state}.",
            "To fix the {err}, update the {arg} setting and restart the {svc} service.",
            "Each call to {fn} must pass a valid {arg}; otherwise the {svc} service raises a {err}.",
            "The {svc} module caches the {arg} so that {fn} is not called twice and still returns {ret}.",
        ],
        {
            "fn": "parse_config load_index send_batch open_session merge_rows build_cache".split(),
            "ret": ["an empty list", "None", "a status code", "the cached value", "a new handle"],
            "arg": "timeout path token buffer schema locale".split(),
            "state": ["missing", "empty", "out of range", "already closed"],
            "err": ["timeout error", "permission error", "key error", "memory leak", "race condition"],
            "svc": "auth storage billing search logging".split(),
        },
    ),
    "medical": (
        [
            "Patients with {cond} should take {drug} {freq} and report any {symptom} to their physician.",
            "The study found that {drug} reduced {symptom} in adults with {cond}.",
            "If {symptom} persists for more than {dur}, stop the {drug} and seek medical advice.",
            "A history of {cond} increases the risk of {symptom} after treatment with {drug}.",
        ],
        {
            "cond": ["hypertension", "type 2 diabetes", "asthma", "chronic migraine", "kidney disease", "arthritis"],
            "drug": ["the tablet", "the inhaler", "the antibiotic", "the injection", "the ointment"],
            "freq": ["twice daily", "once a day with food", "every eight hours", "before bedtime"],
            "symptom": ["dizziness", "a persistent cough", "swelling", "nausea", "a skin rash", "shortness of breath"],
            "dur": ["two days", "a week", "forty-eight hours", "ten days"],
        },
    ),
    "legal": (
        [
            "The {party} shall {duty} within {term} of receiving written notice from the {other}.",
            "Nothing in this {doc} limits the liability of the {party} for {harm}.",
            "Either the {party} or the {other} may terminate this {doc} if the other party fails to {duty}.",
            "The {party} warrants to the {other} that it will {duty} for the duration of this {doc}.",
        ],
        {
            "party": "tenant supplier licensee employer contractor borrower".split(),
            "other": "landlord purchaser licensor employee client lender".split(),
            "duty": [
                "pay all outstanding fees",
                "return the equipment",
                "remedy the breach",
                "provide the required documents",
                "maintain adequate insurance",
            ],
            "term": ["thirty days", "ten business days", "three months", "one year"],
            "doc": "agreement lease contract licence policy".split(),
            "harm": ["fraud", "gross negligence", "personal injury", "wilful misconduct"],
        },
    ),
    "chat": (
        [
            "Hey, are you still coming to the {event} {time}? {name} said the {thing} might be late.",
            "Honestly I forgot about the {event}, can you remind {name} to bring the {thing}?",
            "No worries, just text me when you leave the {event} and I will bring the {thing} {time}.",
            "Did {name} tell you the {event} moved to {time}? I still have your {thing}.",
        ],
        {
            "event": "party meeting game dinner concert picnic".split(),
            "time": ["tonight", "tomorrow morning", "on Saturday", "after work", "next week"],
            "name": "Sam Alex Maria Jordan Priya Leo".split(),
            "thing": "tickets keys charger umbrella cake laptop".split(),
        },
    ),
    "finance": (
        [
            "The {inst} raised its {rate} by {amt} after {metric} came in above forecasts.",
            "Shares in the {sector} sector fell {amt} as investors reacted to weaker {metric}.",
            "Analysts expect the {inst} to hold its {rate} steady until {metric} improves.",
            "The {sector} sector reported higher {metric}, lifting the index by {amt}.",
        ],
        {
            "inst": ["central bank", "treasury", "regulator", "pension fund", "credit union"],
            "rate": ["benchmark rate", "reserve requirement", "lending rate", "deposit rate"],
            "amt": ["a quarter point", "two percent", "half a point", "five percent", "one percent"],
            "metric": ["inflation", "quarterly earnings", "employment figures", "consumer spending", "export volumes"],
            "sector": "energy banking retail technology housing transport".split(),
        },
    ),
}


class DataBuilder:
    """Builds the four data files (the former main() of the script) and can print a short check of them."""

    def __init__(self, out_dir, wordnet, per_category, wikitext, n_calibration, n_heldout, n_new_domain, seed):
        self.out_dir, self.wordnet, self.per_category, self.wikitext = out_dir, wordnet, per_category, wikitext
        self.n_calibration, self.n_heldout, self.n_new_domain, self.seed = n_calibration, n_heldout, n_new_domain, seed

    def _open(self, name):
        return open(os.path.join(self.out_dir, name), "w", encoding="utf8")

    def write_probe_words(self):
        """probe_words.txt: word<TAB>category, a word is listed once (under its first category)."""
        words = wordnet_words(self.per_category) if self.wordnet else placeholder_words()
        seen = set()
        n = 0
        with self._open("probe_words.txt") as f:
            f.write(
                f"# {'WordNet' if self.wordnet else 'PLACEHOLDER'} probe words: word<TAB>category. The first 8 categories are the core set; the order of first appearance defines it.\n"
            )
            for cat, ws in words.items():
                for w in ws:
                    if w.lower() not in seen:
                        seen.add(w.lower())
                        f.write(f"{w}\t{cat}\n")
                        n += 1
        print(f"probe_words.txt: {n} words in {len(words)} categories")

    def wikitext_sentences(self, split, k, rng):
        """k random sentences (8 to 40 words, no headings) of a WikiText-103 split."""
        from datasets import load_dataset

        out = []
        for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split):
            for s in re.split(r"(?<=[.!?])\s+", row["text"].strip()):
                if 8 <= len(s.split()) <= 40 and not s.startswith("=") and "@" not in s:
                    out.append(s)
        rng.shuffle(out)
        return out[:k]

    def build(self):
        """Write the four files."""
        os.makedirs(self.out_dir, exist_ok=True)
        rng = random.Random(self.seed)
        self.write_probe_words()
        if self.wikitext:
            calib, held, tag = (
                self.wikitext_sentences("validation", self.n_calibration, rng),
                self.wikitext_sentences("test", self.n_heldout, rng),
                "WikiText-103",
            )
        else:
            g = fill(GENERAL_T, GENERAL_S, self.n_calibration + self.n_heldout, rng)
            calib, held, tag = g[: self.n_calibration], g[self.n_calibration :], "PLACEHOLDER (template-generated)"
        per = -(-self.n_new_domain // len(DOMAINS))
        new = [s for name, (t, s_) in DOMAINS.items() for s in fill(t, s_, per, rng)]
        rng.shuffle(new)
        new = new[: self.n_new_domain]
        for name, lines, note in (
            ("calibration_sentences.txt", calib, tag),
            ("heldout_sentences.txt", held, tag),
            ("new_domain_sentences.txt", new, "PLACEHOLDER (template-generated): code, medical, legal, chat, finance"),
        ):
            with self._open(name) as f:
                f.write(f"# {note}. One sentence per line.\n" + "\n".join(lines) + "\n")
            print(f"{name}: {len(lines)} sentences")

    def check(self):
        """Print the words per category and a few sample lines of each sentence file."""
        import pandas as pd

        pw = pd.read_csv(
            os.path.join(self.out_dir, "probe_words.txt"),
            sep="\t",
            comment="#",
            header=None,
            names=["word", "category"],
        )
        print("words per category (the first 8 are the core set):")
        print(pw.category.value_counts(sort=False).to_frame("words").T.to_string())
        print("\nfirst words per category:")
        print(
            pw.groupby("category", sort=False)
            .word.apply(lambda s: ", ".join(s.head(8)))
            .to_frame("first words")
            .to_string()
        )
        for name in ("calibration_sentences.txt", "heldout_sentences.txt", "new_domain_sentences.txt"):
            lines = open(os.path.join(self.out_dir, name), encoding="utf8").read().splitlines()
            print(f"\n{name}  |  {lines[0]}")
            for s in lines[1:4]:
                print("   ", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--wordnet",
        action="store_true",
        help="take the probe words from WordNet (default: the short built-in placeholder lists)",
    )
    ap.add_argument("--per-category", type=int, default=250, help="words per category (default 250)")
    ap.add_argument(
        "--wikitext",
        action="store_true",
        help="take the calibration and held-out sentences from WikiText-103 (default: templates)",
    )
    ap.add_argument("--n-calibration", type=int, default=200, help="calibration sentences (default 200)")
    ap.add_argument("--n-heldout", type=int, default=500, help="held-out sentences (default 500)")
    ap.add_argument("--n-new-domain", type=int, default=500, help="new-domain sentences (default 500)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the sentence sampling (default 0)")
    ap.add_argument("--out-dir", default="data", help="output folder (default: ./data)")
    ap.add_argument(
        "--check", action="store_true", help="after building, print the words per category and a few sample lines"
    )
    a = ap.parse_args()
    builder = DataBuilder(
        a.out_dir, a.wordnet, a.per_category, a.wikitext, a.n_calibration, a.n_heldout, a.n_new_domain, a.seed
    )
    builder.build()
    if a.check:
        builder.check()


if __name__ == "__main__":
    main()
