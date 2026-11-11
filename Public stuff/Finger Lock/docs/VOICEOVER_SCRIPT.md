# Voice-over script: How my neural net works

This goes with `stop_motion.mp4`, which is 3:00 long and has no sound. Each time stamp is when that caption shows up on screen. Every picture stays up for a few seconds, so it's fine if I run a little behind.

All the numbers in the video come from my real model (`models/finger_cnn.npz`) and one real photo: the 4-finger photo in row 4, column 0 of `sample_photos.png`.

---

## 1. The lock (0:00 – 0:20)

| Time | On screen | What I say |
|---|---|---|
| 0:00 | Title | This is how the neural network in my finger lock works. |
| 0:04 | Box and button | I press the button on the box. |
| 0:06 | LED matrix | The LEDs light up three slots, one for each number of the code. |
| 0:08 | Camera and hand | Then I hold up my fingers under the camera. |
| 0:11 | Red square | The program only uses the inside of the red square. |
| 0:13 | 64 × 64 image | It shrinks that down to a 64 by 64 color image. |
| 0:15 | Grid of numbers | To the computer, the picture is just numbers between 0 and 1. |
| 0:18 | R, G, B sheets | There are three colors, so that's 3 times 64 times 64. That's 12,288 numbers. |

## 2. Convolution (0:20 – 0:54)

| Time | On screen | What I say |
|---|---|---|
| 0:20 | Filter weights | The first layer is a convolution. A filter is a tiny 3 by 3 grid of weights, with one grid for each color. These are the real weights from my network. |
| 0:25 | Window on the image | The filter sits on one 3 by 3 patch of the image. It multiplies each pixel by its weight. Then it adds everything up, plus a bias. |
| 0:29 | Big positive answer | If the patch looks like the filter's pattern, the answer is big. |
| 0:33 | Negative answer | If the patch looks like the opposite pattern, the answer is negative. |
| 0:37 | Map filling in | Then the filter slides across the whole image. That's 4,096 spots, and it uses the same 28 numbers every time. All the answers together make a feature map. |
| 0:42 | Finished map | This filter lights up on the edges of my fingers. |
| 0:46 | 16 maps | Layer 1 has 16 filters, so it makes 16 feature maps. |
| 0:50 | 16 maps | Each one looks for a different pattern. Red means positive and blue means negative. |

## 3. ReLU (0:54 – 1:09)

| Time | On screen | What I say |
|---|---|---|
| 0:54 | Map and numbers | Some of the numbers in a feature map are negative. Those are the blue ones. |
| 0:58 | ReLU graph | Next is ReLU. If a number is negative, it becomes 0. |
| 1:00 | Numbers flipping to 0 | If it's positive, it stays the same. |
| 1:05 | "Same ReLU as my V data fit" | This is the same ReLU I used in my V data fit. It's just the max of 0 and z. |

## 4. Max pooling (1:09 – 1:24)

| Time | On screen | What I say |
|---|---|---|
| 1:09 | 4 × 4 numbers | Then comes max pooling. It looks at each 2 by 2 block of numbers. |
| 1:11 | Blocks, one at a time | It keeps only the biggest number and throws away the other three. |
| 1:19 | 64 × 64 → 32 × 32 | So the map shrinks from 64 by 64 down to 32 by 32. |
| 1:21 | All 16 maps | This also means that if my finger moves over a few pixels, it still looks about the same. |

## 5. Going deeper (1:24 – 1:48)

| Time | On screen | What I say |
|---|---|---|
| 1:24 | Stacks of maps | Every layer does the same three steps: convolution, ReLU, and max pooling. |
| 1:27 | Filter counts | Width means how many filters a layer has. Mine go 16, then 32, then 64. |
| 1:30 | Shapes | At each layer, the maps get smaller, but there are more of them. |
| 1:32 | Layer 2 maps | Layer 2 makes 32 maps that are 16 by 16. |
| 1:35 | Layer 3 maps | Layer 3 makes 64 maps that are 8 by 8. |
| 1:37 | Layer 3 maps | The deeper maps look blurry to us, but they mean more to the network. |
| 1:40 | Stack of layers | Depth means how many layers are stacked. Mine has 3 conv layers and 1 linear layer. |
| 1:43 | "edges → fingers" | The first layer finds simple things like edges. The later layers find bigger things, like whole fingers. |

## 6. The answer (1:48 – 2:08)

| Time | On screen | What I say |
|---|---|---|
| 1:48 | 64 × 8 × 8 stack | The last conv layer gives 64 maps that are 8 by 8. |
| 1:50 | Long strip | I flatten them into one long line of 4,096 numbers. |
| 1:53 | Linear layer | The linear layer gives each answer a weight for every one of those numbers. That's 24,582 weights. |
| 1:57 | 6 scores | It adds them up to get 6 scores, one for each finger count from 0 to 5. |
| 2:01 | Bars growing | Softmax turns the scores into probabilities that add up to 100 percent. |
| 2:04 | Red bar | For this photo, it says 4 fingers and is 97.5 percent sure. That's right! |

## 7. Training (2:08 – 2:38)

| Time | On screen | What I say |
|---|---|---|
| 2:08 | Loss formula | So how did it learn all those weights? Training starts with the loss, which is how wrong it was. |
| 2:10 | Green example | The loss is minus the natural log of the probability it gave the right answer. For this photo, that's only 0.025. |
| 2:12 | Red example | If it had only said 10 percent, the loss would be 2.30. That's big. |
| 2:16 | Layer boxes | Back-propagation uses the chain rule, just like in my V data fit. |
| 2:18 | Red arrows going back | It sends the error backwards through every layer. |
| 2:21 | "Adam nudges…" | Then gradient descent, with the Adam optimizer, nudges all 48,166 weights a little bit downhill. |
| 2:23 | Original photo | I also used data augmentation. |
| 2:24 | Changed photos | Each time, the same photo gets rotated, shifted, mirrored, or made brighter. |
| 2:28 | "a new version every time" | So it never sees the exact same picture twice, and it can't just memorize them. |
| 2:30 | Loss curve | The loss goes down over 60 passes through all 2,024 photos. |
| 2:32 | Confusion matrix | Most of its mistakes are off by one, like mixing up 3 and 4. |
| 2:35 | Confusion matrix | It gets 93.3 percent right on held-out photos that it never trained on. |

## 8. The lock policy (2:38 – 3:00)

| Time | On screen | What I say |
|---|---|---|
| 2:38 | Hand, bars, hold timer | Last is the lock policy. A number only counts if I hold it for 1 second at 70 percent or more. |
| 2:41 | Empty hand | Then I have to move my hand away, so it sees 0 fingers. |
| 2:43 | Second number | Then I hold up the next number. |
| 2:47 | Third number | And then the third one. |
| 2:50 | Lid opens | If all 3 numbers match my code, it unlocks and the motors lift the lid. |
| 2:53 | Big X | If the code is wrong, it shows an X and the box stays shut. |
| 2:55 | Lockout timer | Three wrong tries in a row locks it for 10 seconds. And if I take more than 20 seconds, it starts over. Thanks for watching! |
