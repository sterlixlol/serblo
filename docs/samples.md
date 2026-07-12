# Serblo — sample generations (FINAL weights)

Verbatim samples from the released checkpoints: **base step 135,000** (val 2.7478, PPL 15.61)
and **Instruct** (SFT step 105). Sampling: temp 0.7, top-k 200, repetition penalty 1.3.
Nothing edited — flaws kept on purpose; see each caption.

## Base model (text continuation)

**Unconditioned — real-entity anchoring at its best** (DW = real outlet, Dejan Jović = real analyst; the interview and quote are entirely invented):

> DW: Kako je to da se u Srbiji uvek smenjuju političari?
> „Srbija i region su već dugo vremena povezani. Neki od njih imaju veoma bliske odnose, poput bivšeg predsednika Tomislava Nikolića ili Aleksandra Vučića", kaže za DW analitičar Dejan Jović. [...]

**Unconditioned — the confabulation ceiling, exhibit A** (every fact is fluent and wrong):

> Poznati glumac i reprezentativac, Kijanu Rivs (41), koji je nedavno napunio 38 godina te trenutno snima film "Zemlja nomada", ponovo će biti deo srpske prestonice [...]

**Local-news register** (invented but pitch-perfect texture, real town, real institution):

> [...] na Stadionu malih sportova, od 18:30 časova počinje podela paketa hrane korisnicima socijalne pomoći. Vasilijević Ivković iz Crvenog krsta Vranje apelovao je [...]

**Croatian register, self-consistent ijekavica** (the model matches dialect to source style):

> Izložba je nastala kao rezultat suradnje Centra za kulturu „Zrin" iz Siska s domaćinima izložbe iz Šibenika - Hrvatskom kinotekom i Hrvatskom udrugom likovnih umjetnika [...]

## Instruct model (chat)

> **>>> Kako se zoveš?**
> Zovem se Serblo, mali jezički model za srpski jezik. Zovu me Serra.

*(Identity holds; at temp 0.7 it occasionally invents nicknames — "Serra", once "Serđo".)*

> **>>> Šta je fotosinteza?**
> Fotosinteza je proces kojim biljke, alge i neke bakterije pretvaraju svetlosnu energiju u hemijsku energiju. [...]

> **>>> Koji je glavni grad Srbije?**
> Glavni grad Srbije je Beograd. To je ujedno i najveći grad u zemlji, sa preko 22 miliona stanovnika po zadnjem popisu iz 2011. godine.

*(Correct capital; population off by 21 million. The 125M knowledge ceiling in one sentence.)*

> **>>> Gde si bio juče, brate?**
> Evo me u kafani "Kod Karađorđevića". Šta ima kod mene, jebote. Ništa posebno, ali ko zna gde bih sad otišao da ne poludim od posla i obaveza. Ti kako si, kaži šta ti treba. Nadam se da će nam biti lepo! 💛

*(The conversational-register thesis, demonstrated: authentic kafanski voice, profanity and all.)*

> **>>> Šta sve umeš?**
> Ma svašta, brate. Od saveta do mudrosti - samo reci ti! 💛

Full unedited transcript: `docs/samples_raw.txt`.
