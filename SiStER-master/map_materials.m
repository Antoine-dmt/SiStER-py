% map_materials

figure
fastscatter(xm/1e3,ym/1e3,im,'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
title('MATERIALS')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
