



def line_pass(visdom, x, y, color, win, name, title):
	if win is None or not visdom.win_exists(win, None):
		update = False
	else:
		update = True
	trace = dict(x=[x], y=[y], mode="lines", type='custom', line=dict(color=color), name=name)
	layout = dict(title=title, margin=dict(l=35, r=5, b=25, t=25, pad=4))
	visdom._send({'data': [trace], 'layout': layout, 'win': win, 'name': name, 'append': update}, endpoint = 'update' if update else 'events')


def show_gradient_norm(visdom, model, win, name):
	X = []
	rownames = []
	for n, f in model.named_parameters():
		norm = f.grad.norm().to('cpu')
		X.append(norm)
		rownames.append(n)

	visdom.bar(X=X, win=win,
        opts=dict(
	        title=name,
            rownames=rownames,
	        marginleft=35,
	        marginright=15,
	        margintop=25,
	        marginbottom=35
        ))


