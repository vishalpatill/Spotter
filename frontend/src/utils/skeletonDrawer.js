const CONNECTIONS = [[11,12],[11,23],[12,24],[23,24],[11,13],[13,15],[12,14],[14,16],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32]];
const JOINTS = [11,12,13,14,15,16,23,24,25,26,27,28];

export function drawSkeleton(ctx, landmarks, width, height, isCorrect) {
  if (!landmarks || landmarks.length < 33) return;
  const color = isCorrect ? "#22c55e" : "#ef4444";
  ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 8;
  for (const [i,j] of CONNECTIONS) {
    const a = landmarks[i], b = landmarks[j];
    if (!a || !b) continue;
    ctx.beginPath(); ctx.moveTo(a.x*width, a.y*height); ctx.lineTo(b.x*width, b.y*height); ctx.stroke();
  }
  ctx.shadowBlur = 0;
  for (const idx of JOINTS) {
    const lm = landmarks[idx]; if (!lm) continue;
    ctx.beginPath(); ctx.arc(lm.x*width, lm.y*height, 8, 0, 2*Math.PI); ctx.fillStyle = color; ctx.fill();
    ctx.beginPath(); ctx.arc(lm.x*width, lm.y*height, 4, 0, 2*Math.PI); ctx.fillStyle = "#fff"; ctx.fill();
  }
}
