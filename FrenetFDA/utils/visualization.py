from matplotlib.pyplot import legend
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc
import plotly.io as pio
from plotly.subplots import make_subplots

layout = go.Layout(
    plot_bgcolor='rgba(0,0,0,0)'
)

""" Set of functions for visualization of mean shape, mean curvature, torsion, etc (visualize 3D curves or 2D curves). """

color_list_mean = px.colors.qualitative.Plotly
# dict_color = {"True Mean" : color_list_mean[0], "Arithmetic Mean" : px.colors.qualitative.Set1[6], "SRVF Mean" : px.colors.qualitative.Set1[8], "FS Mean" : px.colors.qualitative.Dark24[5], "Extrinsic Mean" : color_list_mean[2], "Individual Mean" : color_list_mean[3], "True Mean 2" : color_list_mean[1]}
color_list = px.colors.qualitative.Plotly
dict_color = {"True Mean" : color_list_mean[0], "Arithmetic Mean" : color_list_mean[3], "SRVF Mean" : color_list_mean[2], "FS Mean" : color_list_mean[1], "Extrinsic Mean" : color_list_mean[4], "Individual Mean" : color_list_mean[5], "Ref Param" : color_list_mean[0], "Mean Param" : color_list_mean[5]}


def plot_2D(x, y,  legend={"index":False}, mode='lines'):
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, line=dict(width=2, color=color_list[0])))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_array_2D(x, array_y, name_ind, legend={"index":False}, mode='lines'):
    fig = go.Figure(layout=layout)
    N = len(array_y)
    for i in range(N):
        fig.add_trace(go.Scatter(x=x, y=array_y[i], mode=mode, name=name_ind+str(i), line=dict(width=1)))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_array_2D_names(x, array_y, names, legend={"index":False}, mode='lines'):
    fig = go.Figure(layout=layout)
    N = len(array_y)
    for i in range(N):
        fig.add_trace(go.Scatter(x=x, y=array_y[i], mode=mode, name=names[i], line=dict(width=1)))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_2array_2D(x, array_y, legend={"index":False}, mode='lines'):
    fig = go.Figure(layout=layout)
    N = len(array_y)
    for i in range(N):
        fig.add_trace(go.Scatter(x=x[i], y=array_y[i], mode=mode, line=dict(width=1)))
    if legend['index']==True:
        fig.update_layout(
        title=legend["title"],
        xaxis_title=legend["x axis"],
        yaxis_title=legend["y axis"])
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.show()

def plot_3D(features, names, save=False, filename='', mode='lines'):
    fig = go.Figure(layout=layout)
    for i, feat in enumerate(features):
        feat = np.array(feat)
        fig.add_trace(
            go.Scatter3d(
                x=feat[:,0],
                y=feat[:,1],
                z=feat[:,2],
                name=names[i],
                mode=mode,
                line=dict(width=3)
            )
        )
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
                    scene = dict(
                    xaxis = dict(
                        backgroundcolor="rgb(0, 0, 0)",
                        gridcolor="grey",
                        gridwidth=0.8,
                        zeroline=False,
                        showbackground=False,),
                    yaxis = dict(
                        backgroundcolor="rgb(0, 0, 0)",
                        gridcolor="grey",
                        gridwidth=0.8,
                        zeroline=False,
                        showbackground=False,),
                    zaxis = dict(
                        backgroundcolor="rgb(0, 0, 0)",
                        gridcolor="grey",
                        gridwidth=0.8,
                        zeroline=False,
                        showbackground=False,),),
                )
    if save==True:
        pio.write_image(fig, filename, format='png')
    fig.show()



def plot_results_curvature(grid, list_tab_curvature, mean_curvature, list_legend, yaxis_legend='', xaxis_legend=''):

    nb_tab = len(list_tab_curvature)
    nb_curves = len(list_tab_curvature[0])

    fig = go.Figure(layout=layout)

    for k in range(nb_tab):
        fig.add_trace(go.Scatter(x=grid, y=list_tab_curvature[k][0], mode='lines', name=list_legend[k], opacity=0.8, line=dict(
                width=2, dash='solid', color=color_list[k+1],),showlegend=True))

    for i in range(1,nb_curves):
        for k in range(nb_tab):
            fig.add_trace(go.Scatter(x=grid, y=list_tab_curvature[k][i], mode='lines', opacity=0.6, line=dict(
                    width=2,dash='solid',color=color_list[k+1],),showlegend=False))
      
    fig.add_trace(go.Scatter(x=grid, y=mean_curvature, mode='lines', name='true', line=dict(width=3, color='black'), showlegend=True))

    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title=xaxis_legend, yaxis_title=yaxis_legend)
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_layout(font=dict(size=20))
    fig.update_layout(autosize=False,width=1050,height=750,)
    fig.show()

    return fig


def plot_compare_results_curvature(grid, list_tab_curvature, mean_curvature, list_legend, yaxis_legend='', xaxis_legend='', k_color=0):
    
    nb_tab = len(list_tab_curvature)
    nb_curves = len(list_tab_curvature[0])

    fig = make_subplots(rows=1, cols=nb_tab, shared_xaxes=True, shared_yaxes=True, horizontal_spacing = 0.04)

    for k in range(nb_tab):
        fig.add_trace(go.Scatter(x=grid, y=list_tab_curvature[k][0], mode='lines', name=list_legend[k], opacity=0.8, line=dict(
                width=2, dash='solid', color=color_list[k_color+k+1],),showlegend=True), row=1, col=k+1)

        for i in range(1,len(list_tab_curvature[k])):
            fig.add_trace(go.Scatter(x=grid, y=list_tab_curvature[k][i], mode='lines', opacity=0.7, line=dict(
                    width=2,dash='solid',color=color_list[k_color+k+1],),showlegend=False), row=1, col=k+1)

        if k==0:
            fig.add_trace(go.Scatter(x=grid, y=mean_curvature, mode='lines', name='true', line=dict(width=3, color='black'), showlegend=True), row=1, col=k+1)
        else:
            fig.add_trace(go.Scatter(x=grid, y=mean_curvature, mode='lines', name='true', line=dict(width=3, color='black'), showlegend=False), row=1, col=k+1)

        fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black', title_text='s', row=1, col=k+1)
        fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black', row=1, col=k+1)

       
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title=xaxis_legend, yaxis_title=yaxis_legend)
    fig.update_layout(go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig.update_layout(font=dict(size=25))
    fig.update_layout(
        autosize=False,
        width=nb_tab*1000,
        height=750,)
    fig.show()

    return fig


def plot_compare_results_curvature_2(list_grid, list_tab_curvature, mean_grid, mean_curvature, list_legend, yaxis_legend='', xaxis_legend='', k_color=0):
    
    nb_tab = len(list_tab_curvature)
    nb_curves = len(list_tab_curvature[0])

    fig = make_subplots(rows=1, cols=nb_tab, shared_xaxes=True, shared_yaxes=True, horizontal_spacing = 0.04)

    for k in range(nb_tab):
        fig.add_trace(go.Scatter(x=list_grid[k], y=list_tab_curvature[k][0], mode='lines', name=list_legend[k], opacity=0.8, line=dict(
                width=2, dash='solid', color=color_list[k_color+k+1],),showlegend=True), row=1, col=k+1)

        for i in range(1,len(list_tab_curvature[k])):
            fig.add_trace(go.Scatter(x=list_grid[k], y=list_tab_curvature[k][i], mode='lines', opacity=0.7, line=dict(
                    width=2,dash='solid',color=color_list[k_color+k+1],),showlegend=False), row=1, col=k+1)

        if k==0:
            fig.add_trace(go.Scatter(x=mean_grid, y=mean_curvature, mode='lines', name='true', line=dict(width=3, color='black'), showlegend=True), row=1, col=k+1)
        else:
            fig.add_trace(go.Scatter(x=mean_grid, y=mean_curvature, mode='lines', name='true', line=dict(width=3, color='black'), showlegend=False), row=1, col=k+1)

        fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black', title_text='s', row=1, col=k+1)
        fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black', row=1, col=k+1)

       
    fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title=xaxis_legend, yaxis_title=yaxis_legend)
    fig.update_layout(go.Layout(plot_bgcolor='rgba(0,0,0,0)'))
    fig.update_layout(font=dict(size=25))
    fig.update_layout(
        autosize=False,
        width=nb_tab*1000,
        height=750,)
    fig.show()

    return fig

# def plot_geodesic(curves, curvatures, torsions, s):
#     k = len(curves)
#     fig1 = go.Figure(layout=layout)
#     for i in range(k):
#         if i==0:
#             fig1.add_trace(go.Scatter(x=s, y=curvatures[i], mode='lines', name=str(i), line=dict(width=4, color=color_list[1])))
#         elif i==k-1:
#             fig1.add_trace(go.Scatter(x=s, y=curvatures[i], mode='lines', name=str(i), line=dict(width=4, color=color_list[2])))
#         else:
#             fig1.add_trace(go.Scatter(x=s, y=curvatures[i], mode='lines', name=str(i), line=dict(width=2, color=color_list[0])))
#     fig1.update_layout(
#         title='Geodesic on curvature',
#         xaxis_title='s',
#         yaxis_title='')
#     fig1.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig1.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')

#     fig2 = go.Figure(layout=layout)
#     for i in range(k):
#         if i==0:
#             fig2.add_trace(go.Scatter(x=s, y=torsions[i], mode='lines', name=str(i), line=dict(width=4, color=color_list[1])))
#         elif i==k-1:
#             fig2.add_trace(go.Scatter(x=s, y=torsions[i], mode='lines', name=str(i), line=dict(width=4, color=color_list[2])))
#         else:
#             fig2.add_trace(go.Scatter(x=s, y=torsions[i], mode='lines', name=str(i), line=dict(width=2, color=color_list[0])))
#     fig2.update_layout(
#         title='Geodesic on torsion',
#         xaxis_title='s',
#         yaxis_title='')
#     fig2.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig2.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')

#     fig3 = go.Figure(layout=layout)
#     for i, feat in enumerate(curves):
#         if i==0:
#             feat = np.array(feat)
#             fig3.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     name=str(i),
#                     mode='lines',
#                     line=dict(width=6,color=color_list[1])
#                 )
#             )
#         elif i==k-1:
#             feat = np.array(feat)
#             fig3.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     name=str(i),
#                     mode='lines',
#                     line=dict(width=6,color=color_list[2])
#                 )
#             )
#         else:
#             feat = np.array(feat)
#             fig3.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     name=str(i),
#                     mode='lines',
#                     line=dict(width=3,color=color_list[0])
#                 )
#             )
#     fig3.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict( xaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="grey", gridwidth=0.8, zeroline=False, showbackground=False,),
#                     yaxis = dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="grey", gridwidth=0.8, zeroline=False, showbackground=False,),
#                     zaxis = dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="grey", gridwidth=0.8, zeroline=False, showbackground=False,),),)

#     fig1.show()
#     fig2.show()
#     fig3.show()


# def plot_geodesic_curves(curves):
#     k = len(curves)
#     fig = make_subplots(rows=1, cols=k, specs=[[{"type": "scatter3d"}]*k],)
#     for i in range(k):
#         if i==0:
#             feat = np.array(curves[i])
#             fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2],name=str(i),mode='lines',line=dict(width=6,color=color_list[1])), row=1, col=i+1)
#             fig.update_scenes(dict(
#                     xaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     yaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     zaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),), row=1, col=i+1)
#         elif i==k-1:
#             feat = np.array(curves[i])
#             fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2],name=str(i),mode='lines',line=dict(width=6,color=color_list[2])), row=1, col=i+1)
#             fig.update_scenes(dict(
#                     xaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     yaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     zaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),), row=1, col=i+1)
#         else:
#             feat = np.array(curves[i])
#             fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2],name=str(i),mode='lines',line=dict(width=4,color=color_list[0])), row=1, col=i+1)
#             fig.update_scenes(dict(
#                     xaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     yaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),
#                     zaxis = dict(backgroundcolor="rgb(0, 0, 0)",gridcolor="white",gridwidth=0.8,zeroline=False,showbackground=False,visible=False,),), row=1, col=i+1)
#     fig.show()





# def plot_2D_arr(array, legend={"index":False}, mode='lines'):
#     fig = go.Figure(layout=layout)
#     N = len(array)
#     for i in range(N):
#         fig.add_trace(go.Scatter(x=array[i][:,0], y=array[i][:,1], mode=mode, line=dict(width=1, color=color_list[(i-9)%9])))
#     if legend['index']==True:
#         fig.update_layout(
#         title=legend["title"],
#         xaxis_title=legend["x axis"],
#         yaxis_title=legend["y axis"])
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()

# def plot_2D(x, y,  legend={"index":False}, mode='lines'):
#     fig = go.Figure(layout=layout)
#     fig.add_trace(go.Scatter(x=x, y=y, mode=mode, line=dict(width=2, color=color_list[0])))
#     if legend['index']==True:
#         fig.update_layout(
#         title=legend["title"],
#         xaxis_title=legend["x axis"],
#         yaxis_title=legend["y axis"])
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()


# def plot_mean_2D(arr_x, arr_y, x_mean, y_mean, legend={"index":False}):
#     fig = go.Figure(layout=layout)
#     N = len(arr_y)
#     for i in range(N):
#         fig.add_trace(go.Scatter(x=arr_x[i], y=arr_y[i], mode='lines', line=dict(width=1, color=color_list[(i-9)%9])))
#     fig.add_trace(go.Scatter(x=x_mean, y=y_mean, mode='lines', name='mean', line=dict(width=2, color=px.colors.qualitative.Dark24[5])))
#     if legend['index']==True:
#         fig.update_layout(
#         title=legend["title"],
#         xaxis_title=legend["x axis"],
#         yaxis_title=legend["y axis"])
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()



# def plot_3D_means_grey(features1, features2, names1, names2, size=[800, 400]):
#     fig = go.Figure(layout=layout)

#     for i, feat in enumerate(features2):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 name=names2[i],
#                 line=dict(
#             width=5,
#             color=dict_color[names2[i]],
#             ),
#             showlegend=False
#             )
#         )

#     for i, feat in enumerate(features1):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 line=dict(
#                 width=2.5,
#                 dash='solid',
#                 color='grey',),
#                 showlegend=False
#             )
#         )
#     fig.update_layout(scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )
#     fig.update_layout(
#     autosize=False,
#     width=size[0],
#     height=size[1],)

#     fig.show()

# def plot_3D_means(features1, features2, names1, names2, path=""):
#     fig = go.Figure(layout=layout)

#     if names1!='' and names1!="":
#         for i, feat in enumerate(features1):
#             feat = np.array(feat)
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     mode='lines',
#                     name=names1+str(i),
#                     line=dict(width=3, color=color_list[(i+5-9)%9])
#                 )
#             )
#     else:
#         for i, feat in enumerate(features1):
#             feat = np.array(feat)
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     mode='lines',
#                     line=dict(width=3, color=color_list[(i+5-9)%9]),
#                     showlegend=False
#                 )
#             )

#     for i, feat in enumerate(features2):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 name=names2[i],
#                 line=dict(
#             width=6,
#             color=dict_color[names2[i]],
#             )
#             )
#         )
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )
#     if path!="":
#         fig.write_html(path+"means.html")
#     fig.show()



def plot_curvatures_grey(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path="", size=[100,100]):
    N = len(kappa)
    n = len(kappa_mean)
    fig = go.Figure(layout=layout)
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), opacity=0.8, line=dict(
                width=1.5,dash='solid',color='grey',),showlegend=False))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3), showlegend=False))
        # fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))

    # fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
    if path!="":
        fig.write_html(path+"kappa.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_layout(
    autosize=False,
    width=size[0],
    height=size[1],)
    fig.show()

    fig = go.Figure(layout=layout)
    for i in range(N):
        fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), opacity=0.8, line=dict(
                width=1.5,dash='solid',color='grey',),showlegend=False))
    for i in range(n):
        fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3),showlegend=False))
        # fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, color=dict_color[names_mean[i]])))

    # fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
    if path!="":
        fig.write_html(path+"tors.html")
    fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    fig.update_layout(
    autosize=False,
    width=size[0],
    height=size[1],)
    fig.show()


# def plot_curvatures(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path=""):
#     N = len(kappa)
#     n = len(kappa_mean)

#     fig = go.Figure(layout=layout)
#     if names1!='' and names1!="":
#         for i in range(N):
#             fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+5-9)%9])))
#     else:
#         for i in range(N):
#             fig.add_trace(go.Scatter(x=s, y=kappa[i], mode='lines', showlegend=False, line=dict(width=2, dash='dot', color=color_list[(i+5-9)%9])))
#     for i in range(n):
#         fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
#     if path!="":
#         fig.write_html(path+"curv.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()

#     fig = go.Figure(layout=layout)
#     if names1!='' and names1!="":
#         for i in range(N):
#             fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', name=names1+str(i), line=dict(width=2, dash='dot', color=color_list[(i+5-9)%9])))
#     else:
#         for i in range(N):
#             fig.add_trace(go.Scatter(x=s, y=tau[i], mode='lines', showlegend=False, line=dict(width=2, dash='dot', color=color_list[(i+5-9)%9])))
#     for i in range(n):
#         fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=0.8, line=dict(width=3, color=dict_color[names_mean[i]])))
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
#     if path!="":
#         fig.write_html(path+"tors.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()



# def plot_curvatures_raket(s, kappa, tau, kappa_mean, tau_mean, names_mean, names1, path):
#     n_subj = kappa.shape[0]
#     n_rept = kappa.shape[1]
#     n = len(kappa_mean)

#     fig = go.Figure(layout=layout)
#     for i in range(n_subj):
#         for j in range(n_rept):
#             fig.add_trace(go.Scatter(x=s, y=kappa[i,j], mode='lines', name=names1+str(i+1), line=dict(width=2, dash='dot', color=color_list[i])))
#     for i in range(n):
#         fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], opacity=1, line=dict(width=3, color=dict_color[names_mean[i]])))
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
#     # fig.update_layout(xaxis_title='s', yaxis_title='curvature')
#     # fig.update_layout(showlegend=False)
#     # fig.write_html(path+"curv.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()

#     fig = go.Figure(layout=layout)
#     for i in range(n_subj):
#         for j in range(n_rept):
#             fig.add_trace(go.Scatter(x=s, y=tau[i,j], mode='lines', name=names1+str(i+1), line=dict(width=2,  dash='dot', color=color_list[i])))
#     for i in range(n):
#         fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], opacity=1, line=dict(width=3, color=dict_color[names_mean[i]])))
#     # fig.update_layout(xaxis_title='s', yaxis_title='torsion')
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
#     # fig.update_layout(showlegend=False)
#     # fig.write_html(path+"tors.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()



# def plot_3D_means_raket(features1, features2, names1, names2, path):
#     fig = go.Figure(layout=layout)

#     n_subj = features1.shape[0]
#     n_rept = features1.shape[1]
#     for j in range(n_subj):
#         for k in range(n_rept):
#             feat = np.array(features1[j,k])
#             if k==0:
#                 fig.add_trace(
#                     go.Scatter3d(
#                         x=feat[:,0],
#                         y=feat[:,1],
#                         z=feat[:,2],
#                         mode='lines',
#                         opacity=0.8,
#                         name=names1+str(j+1),
#                         # showlegend=False,
#                         line=dict(width=2,color=color_list[j])
#                     )
#                 )
#             else:
#                 fig.add_trace(
#                     go.Scatter3d(
#                         x=feat[:,0],
#                         y=feat[:,1],
#                         z=feat[:,2],
#                         mode='lines',
#                         opacity=0.8,
#                         showlegend=False,
#                         line=dict(width=2,color=color_list[j])
#                     )
#                 )

#     for i, feat in enumerate(features2):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 name=names2[i],
#                 line=dict(
#             width=12,
#             color=dict_color[names2[i]],
#             )
#             )
#         )
#     fig.update_layout(
#     legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )

#     fig.write_html(path+".html")
#     fig.show()


# def plot_3D_means_raket_grey(features1, features2, names1, names2, path):
#     fig = go.Figure(layout=layout)

#     n_subj = features1.shape[0]
#     n_rept = features1.shape[1]
#     for j in range(n_subj):
#         for k in range(n_rept):
#             feat = np.array(features1[j,k])
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     mode='lines',
#                     opacity=0.3,
#                     name=names1+str(j)+' ,'+str(k),
#                     showlegend=False,
#                     line=dict(width=1.5,color='grey')
#                 )
#             )

#     for i, feat in enumerate(features2):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 name=names2[i],
#                 line=dict(
#             # width=12,
#             width=5,
#             color=dict_color[names2[i]],
#             )
#             )
#         )
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )

#     fig.write_html(path+".html")
#     fig.show()


# green_color = ['#008158', '#00AD7A', '#00CC96', '#17E098', '#2FF198']
# blue_color = ['#2833A6', '#424DD2', '#636EFA', '#6B90FF', '#76B3FF']
# red_color = ['#A62309', '#D2391E', '#EF553B', '#FE6D43', '#FF874E']

# def plot_means_cond_raket(s, kappa_mean, tau_mean, name):

#     n_cond = len(kappa_mean)
#     kappa_T = [kappa_mean[2+i*3] for i in range(5)]
#     kappa_M = [kappa_mean[1+i*3] for i in range(5)]
#     kappa_S = [kappa_mean[i*3] for i in range(5)]
#     tau_T = [tau_mean[2+i*3] for i in range(5)]
#     tau_M = [tau_mean[1+i*3] for i in range(5)]
#     tau_S = [tau_mean[i*3] for i in range(5)]

#     fig = go.Figure(layout=layout)
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=kappa_T[i], mode='lines', name=name+str(3+i*3), line=dict(width=2,  color=blue_color[i])))
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=kappa_M[i], mode='lines', name=name+str(2+i*3), line=dict(width=2, color=red_color[i])))
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=kappa_S[i], mode='lines', name=name+str(1+i*3), line=dict(width=2, color=green_color[i])))
#     fig.add_trace(go.Scatter(x=s, y=kappa_mean[-1], mode='lines', name=name+str(16), line=dict(width=2, color=px.colors.qualitative.Set2[5])))
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()

#     fig = go.Figure(layout=layout)
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=tau_T[i], mode='lines', name=name+str(3+i*3), line=dict(width=2, color=blue_color[i])))
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=tau_M[i], mode='lines', name=name+str(2+i*3), line=dict(width=2,  color=red_color[i])))
#     for i in range(5):
#         fig.add_trace(go.Scatter(x=s, y=tau_S[i], mode='lines', name=name+str(1+i*3), line=dict(width=2, color=green_color[i])))
#     fig.add_trace(go.Scatter(x=s, y=tau_mean[-1], mode='lines', name=name+str(16), line=dict(width=2, color=px.colors.qualitative.Set2[5])))
#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()


# def plot_3D_means_cond_raket(features, name, path):
#     fig = go.Figure(layout=layout)

#     n_cond = len(features)
#     feat_T = [features[2+i*3] for i in range(5)]
#     feat_M = [features[1+i*3] for i in range(5)]
#     feat_S = [features[i*3] for i in range(5)]

#     for i in range(5):
#         feat = feat_T[i]
#         fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=4,color=blue_color[i])))
#     for i in range(5):
#         feat = feat_M[i]
#         fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=4,color=red_color[i])))
#     for i in range(5):
#         feat = feat_S[i]
#         fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(3+i*3),line=dict(width=4,color=green_color[i])))
#     feat = features[-1]
#     fig.add_trace(go.Scatter3d(x=feat[:,0],y=feat[:,1],z=feat[:,2], mode='lines',name=name+str(16),line=dict(width=4,color=px.colors.qualitative.Set2[5])))
#     fig.update_layout(
#     legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )

#     fig.write_html(path+".html")
#     fig.show()



# def plot_curvatures_grey_bis(s, kappa, tau, kappa_mean, tau_mean, names_mean, name1, path="", title=""):
#     N = len(kappa)
#     n = len(kappa_mean)

#     mean_kappa = np.mean(kappa, axis=0)
#     min_kappa = np.amin(np.array(kappa), axis=0)
#     max_kappa = np.amax(np.array(kappa), axis=0)
#     color_mean = dict_color[name1]
#     color_mean_rgb = pc.hex_to_rgb(color_mean)
#     color_mean_transp = (color_mean_rgb[0], color_mean_rgb[1], color_mean_rgb[2], 0.2)
#     pc.hex_to_rgb(color_list_mean[0])
#     fig = go.Figure(layout=layout)
#     fig.add_trace(go.Scatter(x=s, y=mean_kappa, mode='lines', name=name1, line=dict(width=3,color=color_mean)))
#     fig.add_trace(go.Scatter(x=s, y=max_kappa, mode='lines', name='Upper Bound', marker=dict(color="#444"),line=dict(width=0),showlegend=False))
#     fig.add_trace(go.Scatter(x=s, y=min_kappa, mode='lines', name='Lower Bound', marker=dict(color="#444"),line=dict(width=0),fillcolor='rgba'+str(color_mean_transp),fill='tonexty',showlegend=False))
#     for i in range(n):
#         if names_mean[i]=='Mean Param':
#             fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, dash='dash', color=dict_color[names_mean[i]])))
#         else:
#             fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, dash='dashdot', color=dict_color[names_mean[i]])))
#         # fig.add_trace(go.Scatter(x=s, y=kappa_mean[i], mode='lines', name=name[i], line=dict(width=3, color=dict_color[names_mean[i]])))
#     if title=="":
#         fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
#     else:
#         fig.update_layout(title=title, legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='curvature')
#     if path!="":
#         fig.write_html(path+"kappa.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()

#     mean_tau = np.mean(tau, axis=0)
#     min_tau = np.amin(np.array(tau), axis=0)
#     max_tau = np.amax(np.array(tau), axis=0)
#     fig = go.Figure(layout=layout)
#     fig.add_trace(go.Scatter(x=s, y=mean_tau, mode='lines', name=name1, line=dict(width=3,color=color_mean)))
#     fig.add_trace(go.Scatter(x=s, y=max_tau, mode='lines', name='Upper Bound', marker=dict(color="#444"),line=dict(width=0),showlegend=False))
#     fig.add_trace(go.Scatter(x=s, y=min_tau, mode='lines', name='Lower Bound', marker=dict(color="#444"),line=dict(width=0), fillcolor='rgba'+str(color_mean_transp),fill='tonexty',showlegend=False))
#     for i in range(n):
#         if names_mean[i]=='Mean Param':
#             fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, dash='dash', color=dict_color[names_mean[i]])))
#         else:
#             fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=names_mean[i], line=dict(width=3, dash='dashdot', color=dict_color[names_mean[i]])))
#             # fig.add_trace(go.Scatter(x=s, y=tau_mean[i], mode='lines', name=name[i], line=dict(width=3, color=dict_color[names_mean[i]])))
#     if title=="":
#         fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
#     else:
#         fig.update_layout(title=title, legend=dict(orientation="h",yanchor="top",y=1.15,xanchor="right", x=1), xaxis_title='s', yaxis_title='torsion')
#     if path!="":
#         fig.write_html(path+"tors.html")
#     fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
#     fig.show()



# def plot_3D_res_simu_method(features, X_mean, title, color):
#     fig = go.Figure(layout=layout)
#     for i, feat in enumerate(features):
#         feat = np.array(feat)
#         fig.add_trace(
#             go.Scatter3d(
#                 x=feat[:,0],
#                 y=feat[:,1],
#                 z=feat[:,2],
#                 mode='lines',
#                 opacity=0.6,
#                 line=dict(width=2, color=color),
#                 showlegend=False
#             )
#         )
#     feat = np.array(X_mean)
#     fig.add_trace(
#         go.Scatter3d(
#             x=feat[:,0],
#             y=feat[:,1],
#             z=feat[:,2],
#             mode='lines',
#             name="True Mean",
#             line=dict(
#         width=6,
#         color=dict_color["True Mean"],
#         )
#         )
#     )
#     fig.update_layout(title=title,
#                     legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                             backgroundcolor="rgb(0, 0, 0)",
#                             gridcolor="grey",
#                             gridwidth=0.8,
#                             zeroline=False,
#                             showbackground=False,),
#                     yaxis = dict(
#                             backgroundcolor="rgb(0, 0, 0)",
#                             gridcolor="grey",
#                             gridwidth=0.8,
#                             zeroline=False,
#                             showbackground=False,),
#                     zaxis = dict(
#                             backgroundcolor="rgb(0, 0, 0)",
#                             gridcolor="grey",
#                             gridwidth=0.8,
#                             zeroline=False,
#                             showbackground=False,),),
#                     )
#     fig.show()


# def plot_3D_res_simu(features_FS, features_SRVF, features_Arithm, X_mean):

#     plot_3D_res_simu_method(features_FS, X_mean, "Results of estimation with Frenet Serret Method", dict_color["FS Mean"])
#     plot_3D_res_simu_method(features_SRVF, X_mean, "Results of estimation with SRVF Method", dict_color["SRVF Mean"])
#     plot_3D_res_simu_method(features_Arithm, X_mean, "Results of estimation with Arithmetic Method", dict_color["Arithmetic Mean"])


# def plot_clusters(clust_data, name):

#     fig = go.Figure(layout=layout)
#     N = len(clust_data)

#     for i in range(N):
#         ni = len(clust_data[i])
#         clust_data_bis_i = centering_set(clust_data[i])
#         ci = color_list[i%9]
#         for j in range(ni):
#             feat = np.array(clust_data_bis_i[j])
#             fig.add_trace(
#                 go.Scatter3d(
#                     x=feat[:,0],
#                     y=feat[:,1],
#                     z=feat[:,2],
#                     mode='lines',
#                     # name=name+str(i),
#                     line=dict(width=3, color=ci),
#                     showlegend=False,
#                 )
#             )

#     fig.update_layout(legend=dict(orientation="h",yanchor="top",y=1.2,xanchor="right", x=1),
#                     scene = dict(
#                     xaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     yaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),
#                     zaxis = dict(
#                          backgroundcolor="rgb(0, 0, 0)",
#                          gridcolor="grey",
#                          gridwidth=0.8,
#                          zeroline=False,
#                          showbackground=False,),),
#                   )
#     fig.show()
