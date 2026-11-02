# Lab 04 - reproducible synthetic mini-TTS experiment (stdlib + NumPy)
from pathlib import Path
import math, wave, csv, json, time, random
import numpy as np
from xml.sax.saxutils import escape
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'lab4_data'; UNITS=DATA/'units'; OUT=ROOT/'audio_output'; FIG=ROOT/'figures'
for p in (UNITS,OUT,FIG,DATA/'hmm_demo'): p.mkdir(parents=True,exist_ok=True)
SR=16000; RNG=np.random.default_rng(457)
WORDS=['moi','khach','hang','so','khong','mot','hai','ba','bon','nam','sau','bay','tam','chin','den','quay']
DIGIT={'0':'khong','1':'mot','2':'hai','3':'ba','4':'bon','5':'nam','6':'sau','7':'bay','8':'tam','9':'chin'}
FREQ={'moi':(210,700),'khach':(150,520),'hang':(180,630),'so':(230,850),'khong':(170,570),'mot':(200,720),'hai':(260,900),'ba':(190,690),'bon':(160,600),'nam':(220,780),'sau':(250,940),'bay':(200,740),'tam':(180,670),'chin':(240,880),'den':(175,650),'quay':(215,820)}
def norm(text):
    out=[]
    for t in ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text).split(): out.extend(DIGIT[c] for c in t) if t.isdigit() else out.append(t)
    return out
def synth(word,variant=0,dur=None):
    d=dur or (0.28+0.025*(len(word)%4)+0.015*variant); n=int(SR*d); t=np.arange(n)/SR
    f1,f2=FREQ[word]; f0=145+9*variant+10*np.sin(2*np.pi*t/max(d,.1)); phase=2*np.pi*np.cumsum(f0)/SR
    # Explicitly synthetic harmonic/formant-like signal; not recorded speech.
    env=np.minimum(1,t/.025)*np.minimum(1,(d-t)/.035); env=np.maximum(0,env)
    y=sum((1/k)*np.sin(k*phase+0.1*k) * (0.55*np.exp(-((k*f0-f1)/230)**2)+0.35*np.exp(-((k*f0-f2)/340)**2)+.10) for k in range(1,13))
    y*=env*(.75+.12*np.sin(2*np.pi*3*t)); y+=RNG.normal(0,.002,n)*env
    y/=max(np.max(np.abs(y)),1e-9); y*=.18*(1-.06*variant)
    return y.astype(np.float32)
def writewav(path,x):
    x=np.clip(x,-.999,.999); b=(x*32767).astype('<i2').tobytes()
    with wave.open(str(path),'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(b)
def readwav(path):
    with wave.open(str(path),'rb') as w: return np.frombuffer(w.readframes(w.getnframes()),'<i2').astype(float)/32768
# 16 tokens, two candidate WAVs each
rows=[]; inventory={}
for word in WORDS:
    inventory[word]=[]
    for v in range(2):
        path=UNITS/f'{word}_{v+1:02}.wav'; x=synth(word,v); writewav(path,x)
        inventory[word].append(path); rows.append([word,f'sim_speaker_{v+1}',len(x)/SR,float(np.sqrt(np.mean(x*x))),str(path.relative_to(ROOT)),'synthetic_harmonic'])
with open(DATA/'metadata.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['label','speaker','duration_s','rms','file_path','source']); w.writerows(rows)
def joincost(a,b):
    n=min(int(.025*SR),len(a),len(b)); nfft=512
    A=np.abs(np.fft.rfft(a[-n:]*np.hanning(n),nfft)); B=np.abs(np.fft.rfft(b[:n]*np.hanning(n),nfft))
    return float(np.mean((20*np.log10(A+1e-6)-20*np.log10(B+1e-6))**2))
def xfade(a,b,ms):
    n=min(int(ms*SR/1000),len(a)//3,len(b)//3)
    if n<1:return np.r_[a,b]
    q=np.linspace(0,1,n); return np.r_[a[:-n],a[-n:]*(1-q)+b[:n]*q,b[n:]]
def choose(tokens):
    cand=[[readwav(p) for p in inventory[t]] for t in tokens]; dp=[{i:(0.0,[i]) for i in range(2)}]
    for j in range(1,len(tokens)):
        cur={}
        for b in range(2):
            opts=[(dp[-1][a][0]+joincost(cand[j-1][a],cand[j][b])+abs(len(cand[j][b])/SR-.32)*4,dp[-1][a][1]+[b]) for a in range(2)]
            cur[b]=min(opts,key=lambda z:z[0])
        dp.append(cur)
    best=min(dp[-1].values(),key=lambda z:z[0]); return [cand[j][v] for j,v in enumerate(best[1])],best[0]
def assemble(xs,ms):
    y=xs[0]
    for x in xs[1:]: y=xfade(y,x,ms)
    return y
sentence='Moi khach hang so 23 den quay 5.'; tokens=norm(sentence)
xs=[readwav(inventory[t][0]) for t in tokens]; writewav(OUT/'raw_concat.wav',assemble(xs,0))
bxs,cost=choose(tokens); writewav(OUT/'best_concat.wav',assemble(bxs,15))
with open(OUT/'unit_selection_cost.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['candidate_policy','join_cost_sum','crossfade_ms']); w.writerows([['fixed_candidate_1',sum(joincost(a,b) for a,b in zip(xs,xs[1:])),0],['dynamic_programming_2_candidates',cost,15]])
# Prosody variants, synthetic signal only
for factor in [.8,1,1.2]:
    y=assemble(bxs,15); m=max(1,int(len(y)*factor)); idx=np.linspace(0,len(y)-1,m); z=np.interp(idx,np.arange(len(y)),y); writewav(OUT/f'duration_{factor:.1f}.wav',z)
for semi in [-2,0,2]:
    y=assemble(bxs,15); z=np.interp(np.arange(len(y))*2**(semi/12),np.arange(len(y)),y,left=0,right=0); writewav(OUT/f'pitch_{semi:+d}.wav',z)
# observable toy HMM trajectory + illustrative parametric synthesized audio
frames=120; tt=np.arange(frames)*.01; f0=145+18*np.sin(2*np.pi*tt/1.2); formant=650+120*np.sin(2*np.pi*tt/2.3)
np.savetxt(DATA/'hmm_demo'/'trajectory.csv',np.c_[tt,f0,formant],delimiter=',',header='time_s,f0_hz,formant1_hz',comments='')
yh=np.concatenate([synth(t,0,.18) for t in tokens]); writewav(OUT/'hmm_tts.wav',yh)
# No pretrained offline neural model is available; do not label a fake result as neural audio.
(DATA/'neural_tts_backend.py').write_text('"""Optional offline neural TTS adapter. No pretrained model is bundled in this lab.\ndef synthesize(text):\n    raise RuntimeError("Install and configure an approved offline pretrained TTS model before running.")\n"""\n',encoding='utf-8')
# SVG plots (self-contained; no plotting dependency)
def svg(path,title,series,xlabel='time (s)',ylabel='amplitude'):
    W,H=900,360; colors=['#2463a6','#c34b3c','#23836c','#8959a8']; parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="30" y="30" font-size="20">{escape(title)}</text><path d="M60 50V300H860" fill="none" stroke="#333"/>']
    for si,(label,x) in enumerate(series):
        a=np.asarray(x); step=max(1,len(a)//800); a=a[::step]; pts=' '.join(f'{60+i*800/max(1,len(a)-1):.1f},{175-float(v)*105/max(np.max(np.abs(a)),1e-8):.1f}' for i,v in enumerate(a)); parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[si%4]}" stroke-width="1.2"/><text x="{680}" y="{65+si*20}" fill="{colors[si%4]}" font-size="13">{escape(label)}</text>')
    parts.append(f'<text x="390" y="340">{escape(xlabel)}</text><text transform="translate(18 210) rotate(-90)">{escape(ylabel)}</text></svg>'); Path(path).write_text(''.join(parts),encoding='utf-8')
svg(FIG/'waveform_join.svg','Join boundary example', [('raw',xs[0]),('smoothed',bxs[0])])
svg(FIG/'hmm_tracks.svg','Toy HMM generation tracks', [('F0',f0/220),('F1',formant/1000)])
svg(FIG/'waveform_outputs.svg','Output waveform comparison', [('raw',readwav(OUT/'raw_concat.wav')),('selected',readwav(OUT/'best_concat.wav')),('HMM-demo',readwav(OUT/'hmm_tts.wav'))])
# Short-time magnitude spectrogram SVG (first 1.2 s of raw concatenation)
y=readwav(OUT/'raw_concat.wav')[:int(1.2*SR)]; win=256; hop=128
frames=np.array([y[i:i+win]*np.hanning(win) for i in range(0,max(1,len(y)-win),hop)])
spec=np.abs(np.fft.rfft(frames,n=512,axis=1)); spec=20*np.log10(spec+1e-6); spec=np.clip(spec,-75,np.max(spec))
rowspec=spec[:,:129]; W,H=840,270; cells=[]
for ti in range(rowspec.shape[0]):
    for fi in range(rowspec.shape[1]):
        v=(rowspec[ti,fi]+75)/max(1,np.max(rowspec)+75); hue=int(240*(1-v)); x=55+ti*780/rowspec.shape[0]; yy=35+(128-fi)*220/128
        cells.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{780/rowspec.shape[0]+.2:.2f}" height="1.8" fill="hsl({hue},75%,{25+45*v:.0f}%)"/>')
(FIG/'spectrogram_join.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330"><rect width="100%" height="100%" fill="white"/><text x="30" y="24" font-size="18">Raw concatenation log-magnitude spectrogram, 0-4 kHz</text>'+''.join(cells)+'<text x="390" y="315">time (s)</text><text x="8" y="150">frequency</text></svg>',encoding='utf-8')
# duration / RMS histogram SVG
hist=[]
for label,vals in [('duration',[r[2] for r in rows]),('RMS',[r[3]*10 for r in rows])]: hist.append((label,vals))
svg(FIG/'inventory_stats.svg','Inventory duration and RMS overview',hist,'unit index','scaled value')
svg(FIG/'join_cost.svg','Join cost by candidate policy', [('fixed',[joincost(a,b) for a,b in zip(xs,xs[1:])]),('selected',[joincost(a,b) for a,b in zip(bxs,bxs[1:])])],'join index','spectral cost')
with open(OUT/'experiment_summary.json','w') as f: json.dump({'sentence':sentence,'tokens':tokens,'selected_total_join_cost':cost,'crossfade_ms':15,'sample_rate':SR,'channels':1,'bit_depth':16,'audio_source':'synthetic harmonic/formant-like signals; not human speech','neural_model_available':False},f,indent=2)
# concise report PDF with 6 designed pages
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics; from reportlab.pdfbase.ttfonts import TTFont
fonts=Path('C:/Windows/Fonts'); pdfmetrics.registerFont(TTFont('ArialUnicode',str(fonts/'arial.ttf'))); styles=getSampleStyleSheet(); styles['Normal'].fontName='ArialUnicode'; styles['BodyText'].fontName='ArialUnicode'; styles['Title'].fontName='ArialUnicode'; styles['Italic'].fontName='ArialUnicode'; story=[]
def page(title,paras,img=None):
    story.append(Paragraph(title,styles['Title'])); story.append(Spacer(1,12))
    for p in paras: story.append(Paragraph(p,styles['BodyText'])); story.append(Spacer(1,8))
    if img: story.append(Paragraph('Biểu đồ minh họa: '+escape(img)+' (SVG trong figures/).',styles['Italic']))
    story.append(PageBreak())
page('LAB 4 - TỔNG HỢP TIẾNG NÓI',['Báo cáo thực nghiệm tái lập được. Câu thử: “Moi khach hang so 23 den quay 5.” Chuỗi token sau chuẩn hóa: '+', '.join(tokens)+'.','Quan trọng: môi trường hiện không có giọng thu âm hay mô hình TTS neural offline. Inventory trong bài là tín hiệu tổng hợp điều hòa có bao phổ giả lập, chỉ dùng để minh họa xử lý số; không thể xem là lời nói tiếng Việt tự nhiên.'], 'inventory_stats.svg')
page('1. Chuẩn hóa và inventory',['Chuẩn hóa chữ thường, bỏ dấu câu và đọc từng chữ số: 23 → hai ba; 5 → năm. Bộ inventory có 16 từ, mỗi từ hai ứng viên WAV mono PCM 16 kHz, 16-bit. metadata.csv lưu nhãn, speaker giả lập, duration, RMS, đường dẫn và nguồn.','Tín hiệu được tạo bằng chuỗi harmonic có F0 biến thiên và hai vùng cộng hưởng, bao biên đầu/cuối và nhiễu nhỏ. Đây là mô hình nguồn-lọc đơn giản, không phải phát âm từ vựng.'], 'inventory_stats.svg')
page('2. Baseline nối thô',['raw_concat.wav nối candidate 1 bằng cách đặt cạnh trực tiếp. Sai khác pha/biên độ tại điểm nối tạo discontinuity; dạng sóng minh họa trong hình.','best_concat.wav dùng dynamic programming trên hai candidate mỗi token. Join cost là MSE log-magnitude phổ trung bình trong 25 ms ở hai phía điểm nối; thêm phạt duration đơn giản. Sau chọn unit áp dụng cross-fade 15 ms.'], 'waveform_join.svg')
page('3. Unit selection và smoothing',['Bảng unit_selection_cost.csv ghi tổng join cost của baseline và đường đi được chọn. Chỉ số phổ trên tín hiệu giả lập không chứng minh naturalness: không có âm vị hay người nghe.','Cross-fade dùng trộn tuyến tính. 0 ms giữ nguyên biên; 5/15/30 ms cần được nghe trên giọng thật để chọn thỏa hiệp giữa click và làm mờ chuyển tiếp. Trong bộ demo này xuất waveform ở 0 và 15 ms; các độ dài khác là tham số có thể tái chạy.'], 'join_cost.svg')
page('4. Prosody và HMM quan sát',['Xuất các biến thể duration 0.8/1.0/1.2 và pitch −2/0/+2 bán âm; pitch shift bằng đổi tốc độ nội suy nên đồng thời đổi duration và phổ, không phải thuật toán pitch-preserving. Vì vậy đây chỉ là minh họa artifact.','hmm_demo/trajectory.csv chứa F0 và F1 theo frame 10 ms của toy HMM. hmm_tts.wav được tạo từ bộ phát tổng hợp toy; đây không phải model HMM đã huấn luyện. HMM thực tế sinh trajectory tham số có ràng buộc động học từ phân bố state, rồi vocoder tạo sóng.'], 'hmm_tracks.svg')
page('5. Neural TTS và đánh giá',['Không tải model ngẫu nhiên từ Internet. neural_tts_backend.py là adapter báo thiếu model; vì thế không tạo neural_tts.wav hay đo latency neural giả. Cần model offline được giảng viên cấp để hoàn tất mục F.','Bộ WAV hiện minh họa đúng định dạng và pipeline số, nhưng không có intelligibility/naturalness MOS hợp lệ. So sánh cần dùng cùng câu, loudness gần nhau, listening test tối thiểu 5 người và ghi ratings.csv. Latency cũng cần đo trên cùng máy.'], 'waveform_outputs.svg')
page('6. Trả lời và kết luận',['1) Unit tự nhiên vẫn nghe kém tự nhiên vì prosody, ngữ cảnh, âm sắc và biên nối không khớp. 2) Target cost đo hợp ngữ cảnh; join cost đo độ êm biên; chỉ tối ưu join có thể chọn unit không phù hợp câu. 3) Cross-fade dài chồng lấn âm vị và làm mờ phụ âm.','4) F0 thuộc kích thích hữu thanh; formant chủ yếu do bộ lọc ống thanh quản. 5) PSOLA đổi khoảng cách pitch epoch; kéo dài hữu thanh quá mức gây buzz. 6) Delta feature làm mượt trajectory nhờ ràng buộc biến thiên. 7) Whole-waveform inventory cần bao phủ ngữ cảnh lớn. 8) Neural học ánh xạ văn bản-tham số/âm thanh, concatenative chọn và ghép mẫu. 9) Cần cố định model/version để tái lập. 10) MOS cao nhưng intelligibility thấp chỉ phù hợp âm thanh nền không cần nội dung chính xác.','Kết luận: các bước inventory, normalization, nối, chọn unit, smoothing và phân tích tham số có thể chạy lại; kết quả chất lượng tiếng nói thực và neural inference còn cần dữ liệu/model được cấp.'])
doc=SimpleDocTemplate(str(ROOT/'Lab04_report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=44); doc.build(story)
print('DONE Lab04; tokens=',','.join(tokens),'join_cost=',cost)



