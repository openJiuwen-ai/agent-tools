/**
 * TurboQuant Π 实现二分开关（cube ApplyPiToQL1 / vec ApplyPiTransposeToRows）。
 * 由 examples/bisect_pi_cube_vec.sh 在编译前改写 0/1，勿手动提交非默认值。
 *
 * A/B：设为 0 可量 ApplyPi 墙钟/MTE2 占比（Π=I 短路语义）。
 */
#ifndef FIA_TURBOQUANT_PI_BISECT_H
#define FIA_TURBOQUANT_PI_BISECT_H

#define TQ_PI_BISECT_CUBE 1
#define TQ_PI_BISECT_VEC 1
// Last-S2 output O=acc@Π: 1 routes through AIC Cube; 0 keeps Vec fallback.
#define TQ_PI_OUTPUT_CUBE 1
// Decode and prefill share the Cube path; raise this threshold only for A/B.
#define TQ_PI_OUTPUT_CUBE_MIN_M 1U

#endif // FIA_TURBOQUANT_PI_BISECT_H
