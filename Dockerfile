# Install dependencies first
FROM python as build
WORKDIR /app
COPY app.py ./
COPY RandomForestClassifier.pkl ./

RUN pip install streamlit pandas numpy scikit-learn


# Use a different base image for the final stage
FROM python
WORKDIR /app
COPY --from=build /app /app

# Install streamlit in the final image
RUN pip install streamlit

# Install streamlit in the final image
RUN pip install scikit-learn

# Add the installation path to the $PATH
ENV PATH="/app:${PATH}"
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
