#!/usr/bin/env bash
# 체크포인트 평가가 끝날 때마다 결과를 푸시하고, epoch 5까지 끝나면 최종 푸시 후 종료.
RUN=/home/latuca/MMDL_solo/runs/sft_general_vic2000_e5
P=/home/latuca/MMDL_solo/scripts/publish_results.sh
LOG=$RUN/publish.log
for EP in 2 4 5; do
  until [ -f "$RUN/.eval_ep$EP.done" ]; do sleep 300; done
  sleep 30
  MSG="sft_general_vic2000_e5: epoch $EP 평가 결과"; [ $EP = 5 ] && MSG="sft_general_vic2000_e5: 전체 완료 (epoch 2/4/5 결과)"
  until bash $P "$MSG" >> $LOG 2>&1; do echo "[$(date '+%F %T')] 푸시 실패 - 10분 후 재시도" >> $LOG; sleep 600; done
  echo "[$(date '+%F %T')] epoch $EP 푸시 완료" >> $LOG
done
