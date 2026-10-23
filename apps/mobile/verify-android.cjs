// Controls the installed debug APK's WebView; screenshots come from Android adb.
const fs=require('node:fs'),path=require('node:path'),{execFileSync}=require('node:child_process');
const {chromium}=require('C:/Users/aljcn/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const adb='D:/.Workspace/android-sdk-opmd/platform-tools/adb.exe';
const out=process.argv[2],stage=process.argv[3]||'setup'; fs.mkdirSync(out,{recursive:true});
const command=(...args)=>execFileSync(adb,['-s','emulator-5554',...args],{encoding:'utf8'}).trim();
const snap=name=>fs.writeFileSync(path.join(out,name+'.png'),execFileSync(adb,['-s','emulator-5554','exec-out','screencap','-p']));
const token=fs.readFileSync(path.resolve(__dirname,'../../../OPMDMobileDemo/pairing-token.txt'),'utf8').trim();
(async()=>{
 command('shell','am','start','-n','org.opmd.followup.demo/.MainActivity');
 const pid=command('shell','pidof','org.opmd.followup.demo');command('forward','tcp:9227','localabstract:webview_devtools_remote_'+pid);
 for(let n=0;n<20;n++){try{const r=await fetch('http://127.0.0.1:9227/json/version');if(r.ok)break}catch{}await new Promise(r=>setTimeout(r,500))}
 const browser=await chromium.connectOverCDP('http://127.0.0.1:9227',{noDefaults:true}); const ctx=browser.contexts()[0],p=ctx.pages()[0];
 const post=async(route,data)=>{const r=await fetch('http://127.0.0.1:8765'+route,{method:'POST',headers:{Authorization:'Bearer '+token,'X-Request-Id':crypto.randomUUID(),'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw new Error(await r.text());return r.json()};
 if(stage==='setup'){
  await p.getByRole('button',{name:'我的',exact:true}).evaluate(el=>el.click()); await p.getByLabel('服务地址').fill('http://10.0.2.2:8765');await p.getByLabel('连接口令').fill(token);await p.getByRole('button',{name:'连接',exact:true}).evaluate(el=>el.click());await p.getByText('已连接电脑',{exact:true}).waitFor();
  const patient=(await post('/api/patients',{alias:'模拟：小陈'})).id;
  const v=await post('/api/visits',{patient_id:patient,scenario:'self_check',site:'右舌侧缘',visit_date:'2026-10-11',symptoms:'模拟：首次就诊准备',payload:{smoking:0}});
  await post('/api/visits/'+v.id+'/submissions',{revision:v.revision});await post('/api/visits/'+v.id+'/reviews',{revision:v.revision,reviewer:'模拟医生',feedback:'模拟：请补一张右侧舌缘照片',requested_items:['photo'],material_note:'模拟：补右侧舌缘照片，注明拍摄日期'});
  await p.getByRole('button',{name:'我的',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'连接',exact:true}).evaluate(el=>el.click());await p.getByText('已连接电脑',{exact:true}).waitFor();await p.getByRole('button',{name:'我的',exact:true}).evaluate(el=>el.click());await p.getByLabel('切换档案').selectOption(patient);await p.getByRole('button',{name:'首页',exact:true}).evaluate(el=>el.click());
  await p.getByRole('button',{name:'去补充资料'}).waitFor(); await p.waitForTimeout(3500);snap('android-home');
  fs.writeFileSync(path.join(out,'scenario.json'),JSON.stringify({patient_id:patient,source_visit:v.id,virtual_camera:true}));
  await p.getByRole('button',{name:'去补充资料'}).evaluate(el=>el.click());await p.getByRole('heading',{name:'照片与资料',exact:true}).waitFor();snap('android-photo-entry');
  await p.getByRole('button',{name:'拍一张'}).evaluate(el=>el.click());await p.waitForTimeout(1000);command('shell','uiautomator','dump','/sdcard/opmd-camera.xml');fs.writeFileSync(path.join(out,'camera-ui.xml'),command('shell','cat','/sdcard/opmd-camera.xml'));
  console.log('APK connected; photo task opened; virtual camera launched');
 } else if(stage==='finish'){
  await p.getByAltText('本次病损照片').waitFor();snap('android-photo-captured');
  await p.getByRole('button',{name:'下一步',exact:true}).evaluate(el=>el.click());snap('android-submit');await p.getByRole('button',{name:'提交给医生'}).evaluate(el=>el.click());await p.getByText('已提交，等待医生查看',{exact:true}).waitFor();
  await p.getByRole('button',{name:'返回',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'我的',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'演示医生视图'}).evaluate(el=>el.click());await p.getByRole('button',{name:/右舌侧缘.*等待医生/}).evaluate(el=>el.click());
  await p.getByLabel('复核人代号').fill('模拟医生');await p.getByLabel('给患者的反馈').fill('模拟：照片已收到，请按约定就诊。');snap('android-doctor-review');
  await p.getByRole('button',{name:'查看研究辅助'}).evaluate(el=>el.click());await p.getByRole('button',{name:'运行真实模型'}).evaluate(el=>el.click());await p.getByText('已完成',{exact:true}).waitFor({timeout:90000});snap('android-research');
  await p.getByRole('button',{name:'返回',exact:true}).evaluate(el=>el.click());if(await p.getByLabel('给患者的反馈').inputValue()!=='模拟：照片已收到，请按约定就诊。')throw new Error('Draft lost');
  await p.getByRole('button',{name:'确认并发送反馈'}).evaluate(el=>el.click());await p.getByText('反馈已保存',{exact:true}).waitFor();snap('android-feedback');
  fs.writeFileSync(path.join(out,'android-acceptance.json'),JSON.stringify({installed_apk:true,native_http_pairing:true,virtual_camera_capture:true,photo_task_closed_loop:true,live_model_from_apk:true,doctor_draft_navigation:true,physical_phone:false,real_doctors:0,real_users:0},null,2));
  console.log('Android photo workflow and live model passed');
 } else if(stage==='restart'){
  await p.getByRole('button',{name:'新增',exact:true}).evaluate(el=>el.click());await p.getByLabel('想告诉医生的情况').fill('模拟：Android重启草稿');await p.waitForTimeout(500);snap('android-keyboard');
  console.log('Patient draft prepared for force-stop');
 } else if(stage==='check-restart'){
  await p.getByRole('button',{name:'新增',exact:true}).evaluate(el=>el.click());if(await p.getByLabel('想告诉医生的情况').inputValue()!=='模拟：Android重启草稿')throw new Error('Android draft lost');snap('android-draft-restored');console.log('Android draft survived force-stop');
 } else if(stage==='report'){
  await p.getByRole('button',{name:'查看待办'}).evaluate(el=>el.click());await p.getByRole('button',{name:'新增',exact:true}).evaluate(el=>el.click());
  const heading=await p.locator('h1').innerText();if(heading==='描述情况')await p.getByRole('button',{name:'下一步',exact:true}).evaluate(el=>el.click());else if(heading==='核对提交')await p.getByRole('button',{name:'上一步',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'添加报告'}).evaluate(el=>el.click());
  await p.waitForTimeout(500);command('shell','uiautomator','dump','/sdcard/opmd-report-picker.xml');fs.writeFileSync(path.join(out,'report-picker.xml'),command('shell','cat','/sdcard/opmd-report-picker.xml'));console.log('Native report picker launched');
 } else if(stage==='report-finish'){
  await p.getByText('synthetic-ui-report.pdf',{exact:true}).waitFor();snap('android-report-selected');
  await p.getByRole('button',{name:'下一步',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'提交给医生'}).evaluate(el=>el.click());await p.getByText('已提交，等待医生查看',{exact:true}).waitFor();
  console.log('Native PDF selection and upload passed');
 } else if(stage==='camera-cancel'){
  await p.getByRole('button',{name:'查看待办'}).evaluate(el=>el.click());await p.getByRole('button',{name:'新增',exact:true}).evaluate(el=>el.click());const h=await p.locator('h1').innerText();if(h==='描述情况')await p.getByRole('button',{name:'下一步',exact:true}).evaluate(el=>el.click());else if(h==='核对提交')await p.getByRole('button',{name:'上一步',exact:true}).evaluate(el=>el.click());await p.getByRole('button',{name:'拍一张'}).evaluate(el=>el.click());console.log('Camera opened for cancellation');
 } else if(stage==='cancel-check'){
  await p.getByRole('heading',{name:'照片与资料',exact:true}).waitFor();await p.getByRole('button',{name:'拍一张'}).waitFor({state:'visible'});snap('android-camera-cancelled');console.log('Camera cancelled; draft and controls retained');
 } else if(stage==='font'){
  await p.getByRole('button',{name:'查看待办'}).evaluate(el=>el.click());
  const scale=command('shell','settings','get','system','font_scale');await p.waitForTimeout(500);
  const size=await p.evaluate(()=>({width:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth}));
  if(size.scrollWidth>size.width+1)throw new Error('Android home overflows '+scale);snap('android-font-'+scale);
  await p.getByRole('button',{name:'新增',exact:true}).evaluate(el=>el.click());const h=await p.locator('h1').innerText();if(h==='照片与资料')await p.getByRole('button',{name:'上一步',exact:true}).evaluate(el=>el.click());
  const form=await p.evaluate(()=>({width:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth}));if(form.scrollWidth>form.width+1)throw new Error('Android form overflows '+scale);snap('android-form-font-'+scale);
  fs.writeFileSync(path.join(out,'font-'+scale+'.json'),JSON.stringify({scale,home:size,form}));console.log('Android font '+scale+' passed');
 }
 await browser.close();
})().catch(e=>{console.error(e);process.exitCode=1});
