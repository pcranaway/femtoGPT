# :robot: femtoGPT

![crates.io](https://img.shields.io/crates/v/femto-gpt.svg)
![GitHub top language](https://img.shields.io/github/languages/top/keyvank/femtoGPT)
![GitHub](https://img.shields.io/github/license/keyvank/femtoGPT)

femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer.

Everything is implemented from scratch, including the tensor processing logic
along with training/inference code of a minimal GPT architecture.

The architecture is very similar/almost identical with Andrej Karpathy's
[nanoGPT video lecture](https://github.com/karpathy/ng-video-lecture).

femtoGPT is a great start for those who are fascinated by LLMs and would like to
understand how these models work in very deep levels.

femtoGPT uses nothing but random generation libraries (`rand`/`rand-distr`), data-serialization
libraries (`serde`/`bincode` for saving/loading already trained models) and a
parallel computing library (`rayon`).

femtoGPT is ~~EXTREMELY SLOW~~ ***relatively fast on CPU 😉***, since most of the primitive operations (E.g Matrix multiplication)
are implemented in the simplest way possible.

Correctness of gradients is checked using gradient-check method, though it still is very
possible that some layers are implemented wrongly. (E.g I'm not sure if my LayerNorm is
bug-free?)

**HELP!** *IF YOU HAVE A COMPUTER WITH PLENTY OF CPUS AND YOU DON'T MIND RUNNING femtoGPT
FOR A FEW HOURS/DAYS, YOU CAN HELP THIS PROJECT A GREAT DEAL! PLZ CONTACT ME :)*

([Discord server](https://discord.gg/wTJFaDVn45) for discussions around the project!)

## Usage

Easy! You'll just need to put the text you want to train your GPT model on, inside
`dataset.txt`. Make sure it has a small number of unique characters! (E.g. the
current dataset has only used 65 different unique characters!)

Then you'll need to run:

```
cargo run --release
```

It will start training the model and will put the training data in the `train_data`
directory. You can stop the training and continue later!

## Output samples

After hours of training on the Shakespeare database, on a 300k parameter model,
this has been the output:

```
LIS:
Tore hend shater sorerds tougeng an herdofed seng he borind,
Ound ourere sthe, a sou so tousthe ashtherd, m se a man stousshan here hat mend serthe fo witownderstesther s ars at atheno sel theas,
thisth t are sorind bour win soutinds mater horengher
```

This is embarrassingly bad, but looking at the bright side, it seems like it has
been able to generate words that are easy to pronounce.

I'm currently training a 10M parameter model to further examine the correctness
of my implementation.

**UPDATE 5th June 2023:**

This has been a new output, after more hours of training on a model with similar scale:

```
What like but wore pad wo me che nogns yous dares,
As supt it nind bupart 'the reed:
And hils not es
```

Obviously the model has started to learn some words and punctuation rules!

**UPDATE 9th June 2023:**

Model was able to reach loss value of ~1.4

Here is an example output:

```
Adistition gone; true; schistoes for mine souls!
Before your home, bariechts should be
Carlam on that's a worf quirer of him so.
What look'd lack away more
To him foot; one hour fortious of saves:
Son;
'Tis all Earl mmistling me.

HARSARTIO:
Why, idless of my mocks fan that were percious.
Having I will thou should and the muour ne'er shor
To purple, when duke thy are out him.
But will bid you doth remember nature.
Even OF hencomey, carniffeit, I joy
Warming; my deed, but it on of mortard,
```

**UPDATE 10th June 2023, sparks of AGI? LOL**

Having trained a 5-layer model on a dataset gathered from Reddit, tokenized with sentencepiece tokenizer with a vocab-size of 500, for about ~10 hours on a 32-core CPU:

Prompt **"I think"**:

```
I think it s not a scientific method of atheism
I think it s not a fairly simple possible to have a
I think that s a possible point, but I m not sure if you re not
I think that s a great idea, but I m not sure how these are you talking
I think it s a simple thing to do, but I m not sure how thes
I think that s a good thing, but I don t think that s what I mean, I think I
I think that s the only way to be able to use it to be in the same way, but
I think that the police are relatively relatively relative
I think that s a simple moron is a simple character
I think that s the only way to go to the second side of the pol
```

Prompt **"Life is"**:

```
Life is not a good idea, but it s not a perfection of the opp
Life is a good thing to say that I don t know what you re talking about, but I don
Life is not the same as a religion, but I m not sure if you re a
Life is a perfectly good job of arguing that you are alm
Life is a perfectly good job of the opposite of the f
Life is a fundamentalist, and then I m not sure how the h
Life is not a good idea, and it s not a perfectly good job, but I
Life is not the same as atheists, but that s the only way to be ac
Life is a bit of a single one of these industry is a f
Life is a good idea to get the opposite of the police offic
```

Prompt **"So sad that"**:

```
So sad that you can tell you what? I think I ve been using it on the scre
So sad that I don t know about it, but I don t think I m not afraid to
So sad that I m not sure if you re not arguing with the fact that you
So sad that I was involved in the future, and I have a few we
So sad that s what I said, I m sure you are almost everything you
So sad that you can do it, and I don t think that the fact that it s a po
So sad that I m not sure if you re arguing with the fact that they are
So sad that s the one too much time, but I m not sure if you re arg
So sad that you are sadly supposed to be a big deal in the world
So sad that I don t know about this, but I m not sure how you can do it, but
```
