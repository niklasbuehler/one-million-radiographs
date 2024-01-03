# %%
import functools
import itertools
import json
import numbers
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydicom
import seaborn as sns
import torch
from IPython import get_ipython
from tqdm import tqdm

while not Path('.toplevel').exists() and Path('..').resolve() != Path().resolve():
    os.chdir(Path('..'))
if str(Path().resolve()) not in sys.path:
    sys.path.insert(0, str(Path().resolve()))


import simonlanger

pd.set_option('max_colwidth', 20)


# %%

basedir = Path('/home/hagerp/neocortex-nas/shared')


# %%

earlycols = ['patientid', 'bodypart', 'pixelarr_dtype',
             'pixelarr_shape', 'pixelarr_non0count']


def prep_dfs(foldercontents_cacheloc, df_loc, df_allfiles_loc, basedir=basedir, from_nas=True):
    if not df_loc.exists():
        if not foldercontents_cacheloc.exists():
            def listitems(d):
                print('rglobbing', d)
                return sorted([x for x in d.rglob('*') if not '/__' in str(x)])

            def listitems_stated(d):
                itemsraw = listitems(d)
                print('stating items in', d)
                res = []
                for curres in simonlanger.multitheaded_map(lambda f: (f, f.stat()), itemsraw):
                    res.append(curres)
                return res

            if from_nas:
                beckenhuefte = listitems_stated(basedir / 'BeckenHüfte')
                gelenke = listitems_stated(basedir / 'Gelenke')

                torch.save([beckenhuefte, gelenke], foldercontents_cacheloc)
            else:
                allitems = listitems_stated(basedir)
                torch.save(allitems, foldercontents_cacheloc)


        if from_nas:
            beckenhuefte, gelenke = torch.load(foldercontents_cacheloc)
            allitems = beckenhuefte + gelenke
        else:
            allitems = torch.load(foldercontents_cacheloc)

        df_allfiles = pd.DataFrame([{'path': f, 'pathstr': str(f), 'stat': s} for (
            f, s) in allitems if stat.S_ISREG(s.st_mode)])
        df_allfiles.set_index('pathstr', inplace=True)
        df_allfiles['path_withoutsquarebrackets'] = df_allfiles.index.str.replace(
            r'\[\d+\]$', r'', regex=True)

        perform_check_squarebrackets = False

        if perform_check_squarebrackets:
            def check_squarebrackets(args):
                path_withoutsquarebrackets, curfiles = args
                assert any(
                    '[' not in f.name for f in curfiles['path']), curfiles

                reslst, resstr = [], ''
                if len(curfiles) <= 1:
                    return reslst, resstr

                for f in curfiles['path']:
                    if str(path_withoutsquarebrackets) == str(f):
                        continue

                    ret = subprocess.run(['cmp', str(path_withoutsquarebrackets), str(
                        f)], capture_output=True, text=True)
                    if ret.returncode != 0:
                        reslst.append(str(f))
                        resstr += (ret.stdout + ret.stderr + '\n').strip()
                return reslst, resstr

            check_squarebrackets_failures = []
            for reslst, resstr in simonlanger.multitheaded_map(check_squarebrackets, list(df_allfiles.groupby('path_withoutsquarebrackets')), threads=0):
                if len(reslst) > 0:
                    check_squarebrackets_failures += reslst
                    print(resstr)

            Path('data/check_squarebrackets_failures.json').write_text(
                json.dumps(check_squarebrackets_failures))

        if not df_allfiles_loc.exists():
            def read_metainfos(pathstr, read_fulldcm=True):
                header = pydicom.dcmread(pathstr, stop_before_pixels=True)
                output_dict = {'header': header, 'errors': ''}

                if read_fulldcm:
                    dcm = pydicom.dcmread(pathstr)
                    if repr(set(dcm.keys()) - set(header.keys())) != '{(7fe0, 0010)}':
                        output_dict['errors'] += f'unusual key difference: set(dcm.keys()) - set(header.keys())\n'

                    try:
                        pixelarr = dcm.pixel_array
                        output_dict['pixelarr_dtype'] = pixelarr.dtype
                        output_dict['pixelarr_shape'] = pixelarr.shape
                        output_dict['pixelarr_min'] = np.min(pixelarr)
                        output_dict['pixelarr_max'] = np.max(pixelarr)
                        output_dict['pixelarr_mean'] = np.mean(pixelarr)
                        output_dict['pixelarr_std'] = np.std(pixelarr)
                        output_dict['pixelarr_non0count'] = np.count_nonzero(
                            pixelarr)
                        pixelarr_non0 = pixelarr[pixelarr != 0]
                        output_dict['pixelarr_non0min'] = np.min(pixelarr_non0)
                        output_dict['pixelarr_non0mean'] = np.mean(
                            pixelarr_non0)
                        output_dict['pixelarr_non0std'] = np.std(pixelarr_non0)
                    except Exception as e:
                        output_dict['errors'] += f'{repr(e)}\n'

                if output_dict['errors'] != '':
                    print(pathstr, output_dict['errors'])

                return pathstr, output_dict

            new_keys = read_metainfos(list(df_allfiles.index)[0])[1].keys()
            df_allfiles = df_allfiles.reindex(
                columns=list(df_allfiles.columns) + list(new_keys))
            # for new_key in new_keys:
            #     df_allfiles[new_keys] = [np.nan] * len(df_allfiles)

            for pathstr, output_dict in simonlanger.multitheaded_map(read_metainfos, list(df_allfiles.index), threads=4):
                for k, v in output_dict.items():
                    if not isinstance(v, numbers.Number):
                        # ease into setting the cellvalue to an object
                        df_allfiles.at[pathstr, k] = ''
                    df_allfiles.at[pathstr, k] = v

            df_allfiles.to_pickle(df_allfiles_loc)
            print('wrote df of len ', len(df_allfiles))
        print('loading df_allfiles')
        df_allfiles = pd.read_pickle(df_allfiles_loc)
        print('loaded df_allfiles')


        # cleanup tool
        print('ensuring no cleanup due')
        planned_for_deletion = {}
        for i, row in tqdm(list(df_allfiles.iterrows())):
            imagetype = []
            try:
                imagetype = row['header'].ImageType
                if 'DRAWING' in imagetype:
                    planned_for_deletion[row['path']] = f'{imagetype=}'
            except AttributeError:
                pass


        if len(planned_for_deletion) > 0:
            for i, row in tqdm(list(df_allfiles.iterrows())):
                if row['path'] not in planned_for_deletion:
                    continue
                    
                ok = False
                for _, innerrow in df_allfiles.iterrows():
                    if innerrow['header'].StudyInstanceUID == row['header'].StudyInstanceUID and innerrow['header'].PatientID == row['header'].PatientID and  innerrow['path'] not in planned_for_deletion:
                        ok = True
                        break
                assert ok


            Path('cleanupscript.sh').write_text(
                '\n'.join(
                    f"rm '{k}'  # reason: {v}"
                    for k, v in planned_for_deletion.items()
                )
            )
            print('cleanup due, cf cleanupscript.sh ; rerun this script after the cleanup and removing the old cache')
            # exit(1)




        # print(df_allfiles)
        # df_allfiles

        # filter out ...[1] etc duplicates

        df = df_allfiles

        beforelen = len(df)
        df = df[df.index == df['path_withoutsquarebrackets']]
        print(
            f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/beforelen:.4f} (square bracket duplicates)')

        beforelen = len(df)
        df = df.iloc[(np.nonzero(df['pixelarr_shape'].fillna(0).to_numpy()))[0]]
        print(
            f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/beforelen:.4f} (no pixel data)')

        # beforelen = len(df)
        # df = df.iloc[(np.nonzero(df['pixelarr_shape'].fillna(0).to_numpy()))[0]]
        # print(f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/len(df_allfiles):.4f} (no pixel data)')

        df['patientid'] = [path.parent.parent.parent.name for path in df['path']]
        df['bodypart'] = [path.parent.parent.parent.parent.name for path in df['path']]

        # expand df columns by all dicom header fields
        dicom_headers_counter = Counter()

        for header in tqdm(list(df['header'])):
            dicom_headers_counter.update(header.keys())

        dcmcols = [f'dcm_{pydicom.datadict.keyword_for_tag(k)}' for k in sorted(
            dicom_headers_counter)]

        df = df.reset_index()
        df = df.reindex(columns=earlycols +
                        dcmcols + [x for x in list(df.columns) if x not in earlycols])

        # df = df.style.hide('header', axis='columns')

        for col in tqdm(dcmcols):
            # print('fetching all ', col)
            tag = pydicom.datadict.tag_for_keyword(col[len('dcm_'):])
            df[col] = [
                header[tag].value if tag in header else np.nan for header in df['header']]

        # FILTER OUT UNUSUAL dcm_ImageType s
        beforelen = len(df)
        df['dcm_ImageType_str'] = [repr(x) for x in df['dcm_ImageType']]
        valcounts = df['dcm_ImageType_str'].value_counts()
        print(valcounts)
        count_images_except_top2imagetypes = valcounts.sum() - valcounts.nlargest(2).sum()
        print('count images, except top 2: ',
              count_images_except_top2imagetypes)
        keep_imagetypes = set(valcounts.nlargest(2).index)

        df_dropimagetypes = df[~df['dcm_ImageType_str'].isin(keep_imagetypes)]
        df = df[df['dcm_ImageType_str'].isin(keep_imagetypes)]

        for i, droprow in df_dropimagetypes.iterrows():
            subdf = df[(df['dcm_StudyInstanceUID'] == droprow['dcm_StudyInstanceUID']) & (
                df['dcm_PatientID'] == droprow['dcm_PatientID'])]
            assert len(subdf) > 0

        print(f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/beforelen:.4f} (unusual image type, all StudyInstanceUID (and PatientID) s remain present)')

        beforelen = len(df)
        df_dropimageshape = df[[x[1] == 650 for x in df['pixelarr_shape']]]
        df = df[[x[1] != 650 for x in df['pixelarr_shape']]]
        for i, droprow in df_dropimagetypes.iterrows():
            subdf = df[(df['dcm_StudyInstanceUID'] == droprow['dcm_StudyInstanceUID']) & (
                df['dcm_PatientID'] == droprow['dcm_PatientID'])]
            assert len(subdf) > 0
        print(f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/beforelen:.4f} (unusual image shape, all StudyInstanceUID (and PatientID) s remain present)')

        beforelen = len(df)
        # 2 or 3 dimensions
        assert all([2 <= len(x) <= 3 for x in df['pixelarr_shape']])
        # rgb if 3 dimensions dimensions
        assert all([x[2] == 3 for x in df['pixelarr_shape'] if len(x) == 3])
        df = df[[len(x) != 3 for x in df['pixelarr_shape']]]
        print(f'dropped {beforelen-len(df)}/{beforelen}={(beforelen-len(df))/beforelen:.4f} (rgb images; all images are 2d now (i.e. no color dimension))')

        df.to_pickle(df_loc)


# %%
foldercontents_cacheloc = Path('data/cache/prep_foldercontents.pt')
df_loc = Path('data/cache/prep_df.pkl')
df_allfiles_loc = Path('data/cache/prep_df_allfiles.pkl')
prep_dfs(foldercontents_cacheloc, df_loc, df_allfiles_loc)

print('reading df.pkl')
df = pd.read_pickle(df_loc)

df


# %%

dcmcols = [x for x in df.columns if x.startswith('dcm_')]

pivotmask = earlycols + dcmcols


# %%


df['dcm_AnatomicRegionSequence_str'] = [','.join((x[(0x8, 0x104)].value for x in seq)
                                                 if not seq != seq else '').lower()
                                        for seq in df['dcm_AnatomicRegionSequence']]


df['dcm_BodyPartExamined_str'] = [(seq if not seq != seq else '').lower()
                                  for seq in df['dcm_BodyPartExamined']]

df_labelcomparison = pd.pivot_table(df, index=['dcm_BodyPartExamined_str'], columns=['bodypart'], values=['patientid'],
                                    sort=False, aggfunc=lambda x: x.count())

df_labelcomparison


# def matplotlib_to_plotly(cmap, pl_entries):
#     h = 1.0 / (pl_entries - 1)
#     pl_colorscale = []

#     for k in range(pl_entries):
#         C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
#         pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

#     return pl_colorscale


# from pydicom.data import get_testdata_files
# filename = get_testdata_files('CT_small.dcm')[0]
# # px.imshow(pydicom.dcmread(filename).pixel_array, color_continuous_scale=matplotlib_to_plotly(plt.cm.bone, 255)).show(renderer='browser')
# plt.imshow(pydicom.dcmread(filename).pixel_array, cmap=plt.cm.bone)


# from pydicom.data import get_testdata_files
# fig = px.imshow(pydicom.dcmread(get_testdata_files('CT_small.dcm')[0]).pixel_array, color_continuous_scale=matplotlib_to_plotly(plt.cm.bone, 255))
# print()



# print('wrote vis/fig.html')

# px.imshow(, , aspect='equal')

# rows =
# cols = rows

# fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(subdf['pathstr'])[:nels]
#                     + [''] * (rows * cols - len(list(subdf['pathstr'])[:nels])))
# for i, path in enumerate(list(subdf['path'])[:nels]):
#     curfig = px.imshow(pydicom.dcmread(path).pixel_array, color_continuous_scale=matplotlib_to_plotly(plt.cm.bone, 255), aspect='equal')
#     layout = curfig.layout
#     for trace in curfig.data:
#         fig.add_trace(trace, row=1 + i // cols, col=1 + i % cols)

# fig.update_layout(title=infostr, coloraxis=dict(colorscale=matplotlib_to_plotly(
#     plt.cm.bone, 255)), scene_aspectratio=dict(x=1, y=1))
# fig.layout.coloraxis = layout.coloraxis
# fig.update_xaxes(**layout.xaxis.to_plotly_json())
# fig.update_yaxes(**layout.yaxis.to_plotly_json())
# fig.show(renderer='browser')

# fig = plt.figure(figsize=(rows, cols))
# for i, path in enumerate(list(subdf['path'])[:nels]):
#     fig.add_subplot(rows, cols, i+1)
#     plt.imshow(pydicom.dcmread(path).pixel_array, cmap=plt.cm.bone)
# plt.savefig('vis/fig.png')

# print('Data point:', x[ind[0]], y[ind[0]])


# get_ipython().run_line_magic('matplotlib', 'inline')


# %%

# subdf = df[df['pixelarr_shape']]


# can be run without already filtering out in the generation of dfs according to imagetype (i.e. previous filtering operation: the no pixel data check)

# df_rgbimgs = df[[len(x) >= 3 for x in df['pixelarr_shape']]].copy()
# print(f'{len(df_rgbimgs)=}')
# dcm = pydicom.dcmread(df_rgbimgs.iloc[0]['path'])

# dcmaaa

# def analyze_imagetypes(df, copytovis=0):
#     print()


#     valcounts = df['dcm_ImageType_str'].value_counts()
#     print(valcounts)
#     print('sum, except top 2: ', valcounts.sum() -  valcounts.nlargest(2).sum())


#     for key, count in valcounts.items():
#         # if count >= 1000:
#         #     continue

#         subdf = df[df['dcm_ImageType_str'] == key]

#         print('\n\n')
#         print(key)
#         print('bodyparts\n', repr(subdf['bodypart'].value_counts()))
#         print('numpaatients', len(set(subdf['patientid'])))

#         # display(df[df['dcm_ImageType_str'] == key])


#         if copytovis:
#             destdir = Path(f'vis/{key}')
#             destdir.mkdir(exist_ok=True)

#             copylst = list(df[df['dcm_ImageType_str'] == key]['path'])
#             import random
#             random.seed(42)
#             random.shuffle(copylst)
#             for i, path in enumerate(tqdm(copylst[:copytovis])):
#                 shutil.copy2(path, destdir)

# # analyze_imagetypes(df_rgbimgs, copytovis=0)

# analyze_imagetypes(df, copytovis=0) # >= 600 --> if applied to the unusual image types, copies all of them to vis


# %%
if False:
    destdir = basedir / '__by_BodyPartExamined'

    mappings_loc = Path('data/BodyPartExamined_mappings.json')

    # mappings = {}
    # for BodyPartExamined_str, subdf in df.groupby('dcm_BodyPartExamined_str'):
    #     BodyPartExamined_str = BodyPartExamined_str.replace(' ', '').lower()
    #     mappings[BodyPartExamined_str] = BodyPartExamined_str
    # mappings_loc.write_text(json.dumps(mappings))

    mappings = json.loads(mappings_loc.read_text())

    # mappings = {}

    df['mapped_BodyPartExamined'] = [mappings[BodyPartExamined_str.replace(
        ' ', '').lower()] for BodyPartExamined_str in df['dcm_BodyPartExamined_str']]

    for mapped_BodyPartExamined, subdf in df.groupby('mapped_BodyPartExamined'):
        mapped_BodyPartExamined = '_' + mapped_BodyPartExamined

        cur_destdir = destdir / mapped_BodyPartExamined
        cur_destdir.mkdir()

        print(cur_destdir, len(subdf))

        for i, path in enumerate(tqdm(list(subdf['path']))):
            try:
                shutil.copyfile(path, cur_destdir / path.name)
            except:
                import time
                time.sleep(2)
                shutil.copyfile(path, cur_destdir / path.name)

    #     BodyPartExamined_str = BodyPartExamined_str.replace(' ', '')
    #     mapped_bodypartexamined = mappings[BodyPartExamined_str]

        # print(BodyPartExamined_str, len(subdf))


# %%

# %%

# df_final = df.copy()
# df_final['oldpath'] = df_final['path']
# df_final['oldpathstr'] = df_final['pathstr']

df_final = df.copy()
vowel_char_map = {ord('ä'): 'ae',
                  ord('ö'): 'oe',
                  ord('ü'): 'ue',
                  ord('Ä'): 'Äe',
                  ord('Ö'): 'Oe',
                  ord('Ü'): 'Ue',
                  ord('ß'): 'ss'}

df_final['oldbodypart'] = df_final['bodypart']
df_final['bodypart'] = [re.sub(r'^DX[-_]', '', x).translate(vowel_char_map).lower()
                        for x in df_final['bodypart']]

destdir = Path('/home/hagerp/idp/dataset')

df_final['oldpath'] = df_final['path']
df_final['oldpathstr'] = df_final['pathstr']

df_final['path'] = [destdir / row['bodypart'] /
                    os.path.relpath(
                        Path(row['path'].parent), row['path'].parent.parent.parent.parent) / row['path'].name
                    for _, row in df_final.iterrows()]
df_final['pathstr'] = [str(path) for path in df_final['path']]


for bodypart, subdf in df_final.groupby('bodypart'):
    print(bodypart)
    for i, row in tqdm(list(subdf.iterrows())):
        try:
            row['path'].parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(row['oldpath'], row['path'])
        except:
            import time
            time.sleep(2)
            row['path'].parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(row['oldpath'], row['path'])

# %%

basedir = destdir
foldercontents_cacheloc = Path('data/cache/prep_foldercontents_local.pt')
df_loc = Path('data/cache/dataset.pkl')
df_allfiles_loc = Path('data/cache/prep_df_allfiles_local.pkl')
prep_dfs(foldercontents_cacheloc, df_loc, df_allfiles_loc, basedir=destdir, from_nas=False)
df = pd.read_pickle(df_loc)
df['oldbodypart'] = df_final['oldbodypart']
df['oldpath'] = df_final['oldpath']
df['oldpathstr'] = df_final['oldpathstr']
pd.to_pickle(df, df_loc)

print('wrote (without findings paths) dataset.pkl to', df_loc, f' with {len(df)=})')

# df_final['stat'] = [path.stat() for path in df_final['path']]

# df_final.to_pickle('data/dataset.pkl')
# print('df_final written to data/dataset.pkl')


# %%

# ADD FINDINGS Paths

df_loc = Path('data/cache/dataset.pkl')
df = pd.read_pickle(df_loc)


# %%

basedir = Path('/home/langers/langer_idp/dataset/')
all_findings = sorted((basedir / '__0_findings').rglob('Befunde_*/*.txt'))

all_findings = dict(zip(simonlanger.ensureunique([re.fullmatch(r'Befund__?(.+)\.txt', x.name).group(1) for x in all_findings]), all_findings))


keyerrors = []
df['findingspath'] = ''
df['findings'] = ''
df['examinationid'] = [x.parent.parent.name for x in df['path']]
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        findingsfile = all_findings[row['examinationid']]
        df.loc[i, 'findingspath'] = findingsfile

        findings = findingsfile.read_text('utf8')
        assert len(findings.strip()) > 0
        df.loc[i, 'findings'] = findings
    except KeyError:
        keyerrors.append(row)


print('NO FINDINGS FOR ', len(keyerrors), ' OF ', len(df))

# %%

pd.to_pickle(df, df_loc)
print('WROTE FINAL DF (including findings)')


# %%
# df = pd.read_pickle(df_loc)

# df_labelcomparison_loc = Path('data/df_labelcomparison.pkl')

# if df_labelcomparison_loc.exists():
#     df_labelcomparison = pd.read_pickle(df_labelcomparison_loc)
#     print('read existing labelcomparison state')


# def plotlyshowimgs(subdf):
#     downsample_factor = 4
#     nels = min(4**2, len(subdf))
#     cols = int(np.ceil(np.sqrt(nels)))

#     def matplotlib_to_plotly(cmap, pl_entries):
#         h = 1.0 / (pl_entries - 1)
#         pl_colorscale = []

#         for k in range(pl_entries):
#             C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
#             pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

#         return pl_colorscale

#     def get_pixels(path):
#         try:
#             return pydicom.dcmread(path).pixel_array
#         except AttributeError:
#             return np.ones((1, 1)) * np.nan

#     pixelarrs = list(simonlanger.multitheaded_map(
#         get_pixels, list(subdf['path'])[:nels], threads=0))
#     combined_pixelarr = np.ones([nels,
#                                 np.max([x.shape[0] for x in pixelarrs]),
#                                 np.max([x.shape[1] for x in pixelarrs])]) * np.nan
#     for i, pixelarr in enumerate(pixelarrs):
#         if len(pixelarr.shape) == 3:
#             print('[WARN]: unusual pixelarr with shape ', pixelarr.shape)
#             pixelarr = pixelarr[:, :, 0]

#         # normalize for visualization here by max value in img, suboptimal since removes physical meaning of values
#         combined_pixelarr[i, :pixelarr.shape[0], :pixelarr.shape[1]] = pixelarr.astype(
#             float) / np.max(pixelarr)

#     combined_pixelarr = combined_pixelarr[:,
#                                           ::downsample_factor,
#                                           ::downsample_factor]
#     print(combined_pixelarr.shape)

#     fig = px.imshow(combined_pixelarr, facet_col=0, facet_col_wrap=cols,
#                     color_continuous_scale=matplotlib_to_plotly(plt.cm.bone, 255))
#     fig.show(renderer='browser')


# def on_pick(showfig, event):
#     artist = event.artist
#     xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
#     # x, y = artist.get_xdata(), artist.get_ydata()
#     ind = event.ind
#     # print('Artist picked:', event.artist)
#     # print('{} vertices picked'.format(len(ind)))
#     # print('Pick between vertices {} and {}'.format(min(ind), max(ind) + 1))
#     # print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))

#     dcmbodypart, bodypart = list(df_labelcomparison.index)[int(
#         ymouse)], list(df_labelcomparison.columns)[int(xmouse)][1]

#     if 'right' in str(event.mouseevent.button).lower():
#         curval = df_labelcomparison.iloc[int(ymouse), int(xmouse)]
#         print(curval)
#         if (curval - int(curval)) > 0.1:
#             df_labelcomparison.iloc[int(ymouse), int(xmouse)] = int(curval)
#         else:
#             df_labelcomparison.iloc[int(ymouse), int(
#                 xmouse)] = int(curval) + 0.5

#         df_labelcomparison.to_pickle(df_labelcomparison_loc)
#         showfig()
#     else:
#         subdf = df[(df['dcm_BodyPartExamined_str'] == dcmbodypart)
#                    & (df['bodypart'] == bodypart)]

#         infostr = f'{dcmbodypart=}, {bodypart=}, {len(subdf)=}'
#         print(infostr, event.mouseevent.button)
#         plotlyshowimgs(subdf)


# def showfig():
#     get_ipython().run_line_magic('matplotlib', 'qt')

#     sns.set(rc={'figure.figsize': (15, 17)})
#     heatmap = sns.heatmap(df_labelcomparison, annot=True, fmt='.1f',
#                           cmap='Blues', picker=True, linecolor='black', linewidths=1)
#     fig = heatmap.get_figure()
#     fig.canvas.callbacks.connect(
#         'pick_event', functools.partial(on_pick, showfig))
#     mngr = plt.get_current_fig_manager()
#     fig.canvas.manager.window.move(0, 0)
#     # geom = mngr.window.geometry()
#     # x,y,dx,dy = geom.getRect()
#     mngr.window.showMaximized()
#     fig.show()


# showfig()


# %%



df_pivot = pd.pivot_table(
    df, index=['bodypart'], sort=False, aggfunc=lambda x: x.count()).T
# df_pivot = df_pivot.reset_index(names='colname')

# df_pivot.sort_values('bodypart')

df_pivot

# %%


fig = px.bar(df_pivot, height=5000, orientation='h')
fig.update_layout(barmode='stack', yaxis=dict(
    categoryorder='array', categoryarray=list(df_pivot.index)[::-1]))
fig.write_html('vis/metadata_presence.html')
# fig.show(renderer='browser')

df_pivot_includingtotal = df_pivot.copy()
df_pivot_includingtotal['_TOTAL'] = df_pivot_includingtotal.sum(axis=1).astype(int)
df_pivot_includingtotal = df_pivot_includingtotal.reindex(columns=sorted(df_pivot_includingtotal.columns))

df_pivot_includingtotal.to_csv('vis/metadata_presence_df.csv')
df_pivot_includingtotal.to_html('vis/metadata_presence_df.html')
df_pivot_includingtotal.to_pickle('vis/metadata_presence_df.pkl')

df_pivot_includingtotal

# %%




