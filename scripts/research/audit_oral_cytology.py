"""Read-only audit of downloaded Zenodo oral cytology patches and annotations."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {k: float(v) for k, v in zip(['min', 'p25', 'median', 'p75', 'max'], np.percentile(a, [0,25,50,75,100]))} if len(a) else {}


def csv_out(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def points(geometry):
    coords = geometry.get('coordinates', [])
    if geometry.get('type') == 'Polygon':
        return [p for ring in coords for p in ring]
    if geometry.get('type') == 'MultiPolygon':
        return [p for polygon in coords for ring in polygon for p in ring]
    return []


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    metadata={p.name:read_csv(p) for p in (a.root/'metadata').glob('*.csv')}
    slides={r['slide_id']:r for r in metadata['slide_metadata.csv']}
    patients={r['patient_id']:r for r in metadata['patient_metadata.csv']}
    patch_rows=metadata['patch_metadata.csv']
    image_paths={p.stem:p for p in (a.root/'patches').glob('*.png')}
    annotations={p.stem:p for p in (a.root/'patch-geojsons').glob('*.geojson')}
    manifest=read_csv(a.root/'manifests/manifest-main.csv')
    manifest_images=[r for r in manifest if r['file-type']=='WSI Patch']
    aliases={};resolved=set()
    declared={r['patch-name'] for r in patch_rows}
    image_counts=Counter(); nuclear_counts=Counter(); sizes=Counter(); modes=Counter(); bad=[];issues=[]
    patient_ids=set(); slide_ids=set(); centers=Counter(); stains=Counter()
    class_stain=defaultdict(Counter); category_patients=defaultdict(set); category_slides=defaultdict(set)
    diagnosis=Counter(); labels_by_patient=defaultdict(set); slide_nuclei=Counter(); slide_patches=Counter()
    raw_hashes=defaultdict(list); pixel_hashes=defaultdict(list); nuclei=[]; patches=[];selected={}
    class_relations=Counter(); mixed=0; total_json_features=0; border=0; invalid=0; patch_nuc_mismatches=[]
    for index,r in enumerate(patch_rows):
        metadata_name=r['patch-name'];name=metadata_name
        if name not in image_paths:
            coordinates='_'.join(name.rsplit('_',2)[-2:])
            matches=[m for m in manifest_images if m['slide-id']==r['wsi-name']
                     and Path(m['file-name']).stem.endswith('_'+coordinates)]
            if len(matches)==1:
                name=Path(matches[0]['file-name']).stem
                aliases[metadata_name]=name
        path=image_paths.get(name);annotation=annotations.get(name)
        if path is None or annotation is None:
            issues.append({'patch':name,'issue':'missing PNG or GeoJSON'});continue
        slide=slides.get(r['wsi-name'])
        if slide is None:
            issues.append({'patch':name,'issue':'unknown slide'});continue
        resolved.add(name)
        category=r['category'];pid=slide['patient_id'];sid=slide['slide_id']
        image_counts[category]+=1;patient_ids.add(pid);slide_ids.add(sid)
        centers[slide['study_site']]+=1;stains[slide['stain_type']]+=1
        class_stain[category][slide['stain_type']]+=1;category_patients[category].add(pid);category_slides[category].add(sid)
        diagnosis[slide['diagnosis_group']]+=1;labels_by_patient[pid].add(category);slide_patches[sid]+=1
        with path.open('rb') as stream:
            raw_hash=hashlib.file_digest(stream,'sha256').hexdigest()
        raw_hashes[raw_hash].append(name)
        try:
            with Image.open(path) as im:
                im.load();width,height=im.size;mode=im.mode
                image_rgb=im.convert('RGB');pixel_hash=hashlib.sha256(image_rgb.tobytes()).hexdigest()
                pixel_hashes[pixel_hash].append(name)
                thumb=image_rgb.copy();thumb.thumbnail((240,240))
                gray=np.asarray(thumb.convert('L'),dtype=np.float32)
                lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
                sharp=float(lap.var());contrast=float(gray.std());white=float((np.asarray(thumb)>240).all(2).mean())
                key=(category,slide['stain_type'])
                if key not in selected:selected[key]=(thumb.copy(),name)
        except Exception as exc:
            bad.append({'patch':name,'error':str(exc)});continue
        sizes[f'{width}x{height}']+=1;modes[mode]+=1
        doc=json.loads(annotation.read_text(encoding='utf-8'));features=doc['features'] if isinstance(doc,dict) else doc
        total_json_features+=len(features);actual=Counter(); slide_nuclei[sid]+=len(features)
        for i,feature in enumerate(features):
            properties=feature.get('properties',{});cl=properties.get('classification',{}).get('name','UNLABELED')
            nuclear_counts[cl]+=1;actual[cl]+=1;class_relations[f'{category}->{cl}']+=1
            xy=points(feature.get('geometry',{}));valid=bool(xy)
            if valid:
                bounds=np.array(xy,dtype=float)[:,:2]
                valid=bool(np.isfinite(bounds).all())
            if valid:
                x0,y0=bounds.min(0);x1,y1=bounds.max(0)
                valid=bool(x1>x0 and y1>y0)
            if not valid:
                invalid+=1;continue
            outside=bool(x0<0 or y0<0 or x1>width or y1>height)
            border+=outside
            nuclei.append({'patch':name,'object_index':i,'patient_id':pid,'slide_id':sid,'stain':slide['stain_type'],
                           'patch_category':category,'nucleus_category':cl,'x0':float(x0),'y0':float(y0),
                           'x1':float(x1),'y1':float(y1),'width':float(x1-x0),'height':float(y1-y0),
                           'outside_image':outside})
        if len(actual)>1:mixed+=1
        if len(features)!=int(r['num-nuclei']):patch_nuc_mismatches.append(name)
        patches.append({'patch':name,'metadata_patch_name':metadata_name,'patient_id':pid,'slide_id':sid,'category':category,'stain':slide['stain_type'],
                        'center':slide['study_site'],'diagnosis_group':slide['diagnosis_group'],'width':width,'height':height,
                        'nuclei':len(features),'mixed_nucleus_labels':len(actual)>1,'sha256':raw_hash,'pixel_sha256':pixel_hash,
                        'thumbnail_laplacian_variance':sharp,'thumbnail_contrast':contrast,'white_pixel_fraction':white})
        if (index+1)%150==0:print(f'Checked {index+1}/{len(patch_rows)} patches',flush=True)
    slide_count_mismatch=[];slide_label_mismatch=[]
    for sid in slide_ids:
        slide=slides[sid]
        if slide_patches[sid]!=int(slide['num_patches']) or slide_nuclei[sid]!=int(slide['num_nuclei']):slide_count_mismatch.append(sid)
    for r in patch_rows:
        if r['wsi-name'] in slides and r['category']!=slides[r['wsi-name']]['category']:slide_label_mismatch.append(r['patch-name'])
    highest=Counter(max(labels_by_patient[p],key=lambda x:['I','II','III','IV'].index(x)) for p in patient_ids)
    missing_manifest=[r['relative-path'] for r in manifest if not (a.root/r['relative-path']).is_file()]
    wsi_docs=[]
    for p in (a.root/'wsi-geojsons').glob('*.geojson'):
        d=json.loads(p.read_text(encoding='utf-8'));fs=d.get('features',[]) if isinstance(d,dict) else d
        wsi_docs.append({'slide_id':p.stem,'features':len(fs),'classified_features':sum('classification' in f.get('properties',{}) for f in fs)})
    overlap_pairs=[]
    by_slide=defaultdict(list)
    for r in patches:
        parts=r['patch'].rsplit('_',2)
        if len(parts)==3 and parts[-1].isdigit() and parts[-2].isdigit():
            by_slide[r['slide_id']].append((r['patch'],int(parts[-2]),int(parts[-1]),r['width'],r['height']))
    for sid,items in by_slide.items():
        for i,x in enumerate(items):
            for y in items[i+1:]:
                dx=min(x[1]+x[3],y[1]+y[3])-max(x[1],y[1]);dy=min(x[2]+x[4],y[2]+y[4])-max(x[2],y[2])
                if dx>0 and dy>0:overlap_pairs.append({'slide_id':sid,'left':x[0],'right':y[0],'overlap_pixels_if_filename_is_top_left':dx*dy})
    source_hashes={p.relative_to(a.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in list((a.root/'metadata').glob('*'))+list((a.root/'manifests').glob('*')) if p.is_file()}
    summary={'source_root':str(a.root),'total_bytes':sum(p.stat().st_size for p in a.root.rglob('*') if p.is_file()),
             'all_metadata_patients':len(patients),'all_metadata_slides':len(slides),'all_slide_category_counts':dict(Counter(r['category'] for r in slides.values())),
             'all_patient_highest_category_counts':dict(Counter(r['representative_category_highest_observed'] for r in patients.values())),
             'available_patches':len(image_paths),'annotated_patches':len(annotations),'available_patch_patients':len(patient_ids),
             'available_patch_slides':len(slide_ids),'patch_category_counts':dict(image_counts),
             'patients_by_patch_category':{k:len(v) for k,v in category_patients.items()},
             'slides_by_patch_category':{k:len(v) for k,v in category_slides.items()},
             'available_patient_highest_patch_category_counts':dict(highest),'patch_stain_counts':dict(stains),
             'patch_category_by_stain':{k:dict(v) for k,v in class_stain.items()},'patch_center_counts':dict(centers),
             'available_patient_centers':dict(Counter(patients[p]['study_site'] for p in patient_ids)),
             'patch_diagnosis_counts':dict(diagnosis),'image_dimensions':dict(sizes),'image_modes':dict(modes),
             'nucleus_annotation_counts':dict(nuclear_counts),'total_nuclei':total_json_features,
             'patch_to_nucleus_label_counts':dict(class_relations),'mixed_nucleus_label_patches':mixed,
             'invalid_geometry_objects':invalid,'out_of_bounds_objects':border,
             'nucleus_width':distribution([r['width'] for r in nuclei]),'nucleus_height':distribution([r['height'] for r in nuclei]),
             'nuclei_per_patch':distribution([r['nuclei'] for r in patches]),'patches_per_patient':distribution([sum(r['patient_id']==pid for r in patches) for pid in patient_ids]),
             'duplicate_file_groups':[v for v in raw_hashes.values() if len(v)>1],
             'duplicate_pixel_groups':[v for v in pixel_hashes.values() if len(v)>1],
             'patient_multi_category_counts':sum(len(v)>1 for v in labels_by_patient.values()),
             'missing_manifest_files':missing_manifest,'manifest_entry_count':len(manifest),
             'filename_aliases_verified_from_official_manifest':aliases,
             'unresolved_metadata_images':sorted(declared-set(image_paths)-set(aliases)),
             'extra_pngs_after_alias_resolution':sorted(set(image_paths)-resolved),
             'extra_annotations_after_alias_resolution':sorted(set(annotations)-resolved),
             'bad_images':bad,'join_issues':issues,'patch_nucleus_count_mismatches':patch_nuc_mismatches,
             'slide_annotation_count_mismatches':slide_count_mismatch,'patch_slide_category_mismatches':slide_label_mismatch,
             'wsi_annotation_files':len(wsi_docs),'wsi_annotation_features':sum(r['features'] for r in wsi_docs),
             'wsi_classified_features':sum(r['classified_features'] for r in wsi_docs),
             'nuclei_with_side_below_8_pixels':sum(r['width']<8 or r['height']<8 for r in nuclei),
             'nuclei_with_side_below_16_pixels':sum(r['width']<16 or r['height']<16 for r in nuclei),
             'possible_overlap_pairs_from_filenames':len(overlap_pairs),'source_table_sha256':source_hashes}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    csv_out(a.output/'patch_inventory.csv',patches);csv_out(a.output/'nucleus_inventory.csv',nuclei)
    csv_out(a.output/'possible_patch_overlaps.csv',overlap_pairs)
    canvas=Image.new('RGB',(800,4*290),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for row,category in enumerate(['I','II','III','IV']):
        for col,stain in enumerate(['PAP','MGG']):
            if (category,stain) not in selected:continue
            thumb,name=selected[(category,stain)]
            x=col*400;y=row*290
            canvas.paste(thumb,(x+10,y+30));draw.text((x+10,y+8),f'Category {category} / {stain}',fill='black')
            draw.text((x+10,y+273),name,fill='black')
    canvas.save(a.output/'category_stain_examples.jpg',quality=90)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_table_sha256'},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
