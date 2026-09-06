# 2026-09-06 — what moves an endpoint's fingerprint, and what does not

Everything the 0.999 threshold of `D-01M1PP6HKHF97G` was set against. Vectors are
normalized, so every figure is a dot product. `gte-modernbert-base` under
llama.cpp `b10809-5266f24da`, one probe sentence unless a row says otherwise.

| What changed | Fingerprint | Verdict |
|---|---|---|
| Nothing — same input twice in one request | 1.0000000000 | kept |
| Nothing — a second server, identical flags | 1.0000000000 | kept |
| Position in a 32-input batch | 0.9999998908 | kept |
| Backend: `-ngl 0` against the default, on Q8_0 | 0.9998787158 | kept |
| Quantization: Q8_0 against F16 | 0.999631 – 0.999883 | kept |
| Quantization: Q8_0 against Q4_k_m | 0.975885 – 0.992211 | **discarded** |
| Different weights entirely | 0.013 | **discarded** |

**The threshold lands where it should, and the margin is thinner than it looks.**
The nearest thing kept is Q8_0 against F16 at 0.999631, six ten-thousandths
above the line; the nearest thing discarded is Q4_k_m at 0.992211, seven
thousandths below. The gap the threshold sits in is about 0.007 wide, not the
seven orders of magnitude between untouched and unrelated.

That the line falls between F16 and Q4_k_m is the right place for it. Adjacent
results in a ranked answer sit about 0.04 apart
([what merging cost and kept](2026-09-06-what-merging-cost-and-kept.md)), so
Q4_k_m's perturbation of up to 0.024 is the same order as the gaps it would have
to preserve, while F16's 0.00037 is two orders below them. One changes what a
query returns and the other does not.

**A second process changes nothing.** Two servers started with identical flags
over the same file return bit-identical vectors, so everything above is
attributable to what the row names and not to running twice.

**The backend row is quantization-dependent**, which
[whether the runtime changes an embedding](2026-09-06-whether-the-runtime-changes-an-embedding.md)
found after this one was written. On F16 the same Metal-against-CPU comparison is 1.5e-6
rather than 1.2e-4, so the figure here is two dequantization paths disagreeing
and not the cost of the backend. It is the right number for Q8_0, which is what
folio ships.

**The fingerprint's sensitivity depends on its text**, which is the known
weakness of the number. For one comparison — Q8_0 against Q4_k_m — an English
sentence reads 0.9913, a line of code 0.9870, and a Korean sentence 0.9759. The
fingerprint folio ships is English, so on a corpus in another language it reports
less movement than that corpus's own content would see. It stayed on the correct
side of the threshold in every case measured here, and it has less room than the
figures suggest.

**Method.** Two extra `llama-server` processes on ports 8081 and 8082, the same
model repository at three quantizations, `--pooling cls -c 8192 -b 8192 -ub 8192`.
`/props` confirmed both servers loaded the same file at the same build for the
backend row. The unrelated-weights row is the stub endpoint described in
[how the name-based rule mixed two models](2026-09-06-how-the-name-based-rule-mixed-two-models.md).
