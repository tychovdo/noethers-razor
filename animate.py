import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.core.display import HTML, display

def animate_trajectory_2d(solution, t_span, n_line_segments=500, H=None, JH=None):
    dim = 2

    fig, ax = plt.subplots(figsize=(4, 4))

    solution = solution.reshape(len(solution), 2, -1, dim)
    n_objects = solution.shape[2] # get n_objects from solution shape
    
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:brown', 'tab:pink']
    
    points = []
    for object_i in range(n_objects):
        point, = ax.plot([], [], 'o', markersize=10, color=colors[object_i % len(colors)])
        points.append(point)
        
    if n_line_segments > 0:
        lines = []
        for object_i in range(n_objects):
            segment = []
            for line_i in range(n_line_segments):

                line, = ax.plot([], [], '-', markersize=10, color=colors[object_i], alpha=1 - (line_i / n_line_segments))
                segment.append(line)
            lines.append(segment)
            
    if (H is not None) or (JH is not None):
        textbox = ax.text(0.5, 0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="center")

    def animate(i):
        for object_i in range(n_objects):
            point_x, point_y = solution[i].reshape(2, n_objects, dim)[0, object_i]
            
            point = points[object_i]
            point.set_data([point_x.item()], [point_y.item()])
            
            if (H is not None) or (JH is not None):
                H_item = H[i].item()
                jH_norm = np.linalg.norm(JH[i]).item()
                jH_max = JH[i].max().item()
                
                textbox.set_text(f'H={H_item:.3f}. Jnorm={jH_norm:.3f}, Jmax={jH_max:.3f}')
                
                
            if n_line_segments > 0:
                line_step = 1
                for line_i in range(n_line_segments):
                    i1 = max(0, i - (line_i + 1) * line_step)
                    i2 = max(0, i - line_i * line_step)

                    line_x1, line_y1 = solution[i1].reshape(2, n_objects, dim)[0, object_i]
                    line_x2, line_y2 = solution[i2].reshape(2, n_objects, dim)[0, object_i]

                    line = lines[object_i][line_i]

                    line.set_data([line_x1, line_x2], [line_y1, line_y2])
                
        return line,

    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    plt.xticks([-3, 0, 3])
    plt.yticks([-3, 0, 3])

    ax.set_aspect('equal')

    ani = FuncAnimation(fig, animate, frames=len(t_span), interval=10, blit=True)

    plt.close()
    
    return ani

def animate_trajectory_3d(solution, t_span, rot_speed=0.2, n_line_segments=150):    
    dim = 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green']

    solution = solution.reshape(len(solution), 2, -1, dim)
    n_objects = solution.shape[2] # get n_objects from solution shape
    
    points = []
    for object_i in range(n_objects):
        point, = ax.plot([], [], [], 'o', markersize=10, color=colors[object_i])
        
        points.append(point)
        
    if n_line_segments > 0:
        lines = []
        for object_i in range(n_objects):
            segment = []
            for line_i in range(n_line_segments):

                line, = ax.plot([], [], '-', markersize=10, color=colors[object_i], alpha=1 - (line_i / n_line_segments))
                segment.append(line)
            lines.append(segment)
            
    def animate(i):
        for object_i in range(n_objects):
            point_x, point_y, point_z = solution[i].reshape(2, n_objects, dim)[0, object_i]
            
            point = points[object_i]
            point.set_data(point_x, point_y)
            point.set_3d_properties(point_z)
                        
            if n_line_segments > 0:
                line_step = 1
                for line_i in range(n_line_segments):
                    i1 = max(0, i - (line_i + 1) * line_step)
                    i2 = max(0, i - line_i * line_step)

                    line_x1, line_y1, line_z1 = solution[i1].reshape(2, n_objects, dim)[0, object_i]
                    line_x2, line_y2, line_z2 = solution[i2].reshape(2, n_objects, dim)[0, object_i]

                    line = lines[object_i][line_i]

                    line.set_data([line_x1, line_x2], [line_y1, line_y2])
                    line.set_3d_properties([line_z1, line_z2])
                    
        ax.view_init(30, 30 + i * rot_speed)

        return points

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_zticks([0])
    
    ax.set_aspect('equal')

    ani = FuncAnimation(fig, animate, frames=len(t_span), interval=10, blit=True)

    plt.close()
    
    return ani

def animate_trajectory(solution, t_span, dim, n_line_segments=500):
    if dim == 2:
        return animate_trajectory_2d(solution, t_span, n_line_segments=n_line_segments)
    elif dim == 3:
        return animate_trajectory_3d(solution, t_span, n_line_segments=n_line_segments)
    else:
        raise NotImplementedError(f"Only supports animations in 2 or 3 dimensions. Got {dim}.")

