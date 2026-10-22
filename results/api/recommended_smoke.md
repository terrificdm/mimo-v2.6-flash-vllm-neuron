# smoke gate: PASS

smoke dir: $MIMO_WORK/results/api/smoke_recommended

| file | HTTP | expect | contains | question |
|---|---|---|---|---|
| single | 200 | None | None | 中国的首都是哪座城市？ |
| single | 200 | None | None | capital of France |
| single | 200 | 391 | True | 17×23=? |
| single | 200 | None | None | 请用两三句话介绍一下长城。 |
| single | 200 | None | None | why the sky is blue |
| single | 200 | 20 | True | 一个数加上它的一半等于 30，这个数是多少？ |
| concurrent4 | 200 | None | None | 请用两三句话介绍一下长城。 |
| concurrent4 | 200 | None | None | why the sky is blue |
| concurrent4 | 200 | 391 | True | 17×23 等于多少？ |
| concurrent4 | 200 | 20 | True | 一个数加上它的一半等于 30，这个数是多少？ |

