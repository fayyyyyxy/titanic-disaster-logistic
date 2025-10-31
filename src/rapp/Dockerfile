FROM rocker/r-ver:4.3.3

WORKDIR /app

# install packages 
COPY src/rapp/install_packages.R /app/install_packages.R
RUN Rscript /app/install_packages.R

# copy the whole repo 
COPY . /app

# run the script using repo-relative path
CMD ["Rscript", "src/rapp/main.R"]